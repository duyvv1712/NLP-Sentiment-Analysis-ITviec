import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold


# Luoi sieu tham so theo dung dac ta ke hoach TV3 (Ngay 2 - Hyperparameter Tuning).
# Logistic Regression tach thanh 2 nhanh vi solver='lbfgs' chi ho tro penalty='l2',
# trong khi 'saga' ho tro ca 'l1' va 'l2'; gop chung vao 1 grid se sinh ra cac to hop
# khong hop le (lbfgs + l1) khien GridSearchCV bao loi/canh bao vo ich.
PARAM_GRIDS: Dict[str, Any] = {
    "Multinomial Naive Bayes": {
        "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
    },
    "Logistic Regression": [
        {"solver": ["lbfgs"], "penalty": ["l2"], "C": [0.1, 1.0, 5.0, 10.0]},
        {"solver": ["saga"], "penalty": ["l1", "l2"], "C": [0.1, 1.0, 5.0, 10.0]},
    ],
    "Support Vector Machine (Linear)": {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
    },
}


class SentimentModelTrainer:
    """
    Quản lý huấn luyện, so sánh và đánh giá các mô hình phân loại cảm xúc.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "Support Vector Machine (Linear)": SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }
        self.trained_models = {}
        self.evaluation_results = []
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.tuning_results: List[Dict[str, Any]] = []

    def get_stacking_model(self, base_estimators: Optional[Dict[str, Any]] = None, cv: int = 5):
        """Tạo Stacking Classifier từ 3 mô hình cơ sở (NB, LR, SVM).

        Nếu `base_estimators` được truyền vào (ví dụ các mô hình đã tune ở bước
        Hyperparameter Tuning), Stacking sẽ dùng đúng cấu hình đó làm base thay
        vì khởi tạo lại tham số mặc định. `cv` là số fold nội bộ StackingClassifier
        dùng để sinh meta-feature (giảm xuống khi cần so sánh nhanh bằng
        cross-validation ở lớp ngoài, tránh CV lồng CV quá tốn thời gian).
        """
        base_estimators = base_estimators or {}
        nb = base_estimators.get("Multinomial Naive Bayes", MultinomialNB())
        lr = base_estimators.get(
            "Logistic Regression",
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        )
        svm = base_estimators.get(
            "Support Vector Machine (Linear)",
            SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
        )
        estimators = [
            ('nb', clone(nb)),
            ('lr', clone(lr)),
            ('svm', clone(svm)),
        ]
        stack_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=cv
        )
        return stack_model

    def tune_hyperparameters(
        self,
        X_train,
        y_train,
        cv: int = 5,
        scoring: str = 'f1_macro',
        random_state: int = 2026,
        n_jobs: int = -1,
        param_grids: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Tìm siêu tham số tối ưu cho từng mô hình cơ sở bằng GridSearchCV.

        SVM được tìm kiếm với `probability=False`: bật Platt scaling trong lúc
        GridSearchCV đã chạy CV bên ngoài sẽ nhân đôi chi phí (CV lồng CV) mà
        không ảnh hưởng đến lựa chọn C tốt nhất. Sau khi chọn được C, mô hình
        cuối cùng được khởi tạo lại với `probability=True` để dùng được
        `predict_proba` (cần cho Stacking).
        """
        grids = param_grids or PARAM_GRIDS
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        self.tuning_results = []
        tuned_models: Dict[str, Any] = {}

        for name, base_model in self.models.items():
            search_model = base_model
            if name == "Support Vector Machine (Linear)":
                search_model = SVC(kernel='linear', class_weight='balanced', probability=False, random_state=42)

            search = GridSearchCV(
                estimator=search_model,
                param_grid=grids[name],
                cv=splitter,
                scoring=scoring,
                n_jobs=n_jobs,
                refit=True,
            )
            print(f"-> Tuning {name}...")
            search.fit(X_train, y_train)

            best_estimator = search.best_estimator_
            if name == "Support Vector Machine (Linear)":
                best_estimator = SVC(
                    kernel='linear',
                    class_weight='balanced',
                    probability=True,
                    random_state=42,
                    **search.best_params_,
                )
                best_estimator.fit(X_train, y_train)

            tuned_models[name] = best_estimator
            self.best_params[name] = search.best_params_
            self.tuning_results.append({
                "Model": name,
                "Best Params": search.best_params_,
                "CV Macro F1 Mean": search.best_score_,
            })

        self.trained_models.update(tuned_models)
        results_df = pd.DataFrame(self.tuning_results).sort_values(
            "CV Macro F1 Mean", ascending=False
        )
        return results_df

    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        score_col: str = "CV Macro F1 Mean",
        save_path: str = None,
    ):
        """Vẽ bar chart so sánh các mô hình theo cột điểm số `score_col`."""
        ordered = results_df.sort_values(score_col, ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=ordered, x=score_col, y="Model", hue="Model", palette="viridis", legend=False)
        plt.title("So sánh hiệu năng các mô hình")
        plt.xlabel(score_col)
        plt.ylabel("")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Đã lưu biểu đồ so sánh tại: {save_path}")
        plt.show()

    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test, target_names: List[str] = None):
        """Huấn luyện và đánh giá toàn bộ danh sách mô hình."""
        self.evaluation_results = []
        
        for name, model in self.models.items():
            print(f"-> Đang huấn luyện mô hình: {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Dự đoán trên tập test
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            self.evaluation_results.append({
                "Model": name,
                "Accuracy": acc,
                "F1 Macro": f1_macro,
                "F1 Weighted": f1_weighted
            })
            
        results_df = pd.DataFrame(self.evaluation_results).sort_values(by="F1 Macro", ascending=False)
        return results_df

    def plot_confusion_matrix(self, model_name: str, y_test, y_pred, labels: List[str], save_path: str = None):
        """Vẽ biểu đồ Confusion Matrix cho mô hình."""
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Dự đoán (Predicted)')
        plt.ylabel('Thực tế (Actual)')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Đã lưu biểu đồ tại: {save_path}")
        plt.show()

    def save_model(self, model_name: str, filepath: str):
        """Lưu mô hình đã huấn luyện."""
        if model_name in self.trained_models:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.trained_models[model_name], filepath)
            print(f"Đã lưu mô hình '{model_name}' tại {filepath}")
        else:
            raise ValueError(f"Mô hình '{model_name}' chưa được huấn luyện.")
