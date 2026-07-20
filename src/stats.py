import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve

# Quy ước: scores_a, scores_b là điểm theo từng fold (đã ghép cặp theo fold_id).
# Với log loss thì NHỎ HƠN = TỐT HƠN, nên diff = scores_a - scores_b < 0 nghĩa là A tốt hơn.


#! Kiểm định ghép cặp 
def paired_ttest(scores_a, scores_b):
    """Paired t-test trên chênh lệch theo từng fold. Trả về (t_stat, p_value)."""
    return stats.ttest_rel(scores_a, scores_b)


def wilcoxon_test(scores_a, scores_b):
    """Wilcoxon signed-rank (phi tham số, dùng kiểm tra chéo khi nghi ngờ phân phối
    không chuẩn). Trả về (stat, p_value); trả p=nan nếu mẫu quá nhỏ/mọi hiệu bằng nhau.
    """
    try:
        return stats.wilcoxon(scores_a, scores_b)
    except ValueError:
        return np.nan, np.nan


#! Khoảng tin cậy cho chênh lệch trung bình (viết cả hai)
def ci_diff_t(scores_a, scores_b, confidence=0.95):
    """CI theo phân phối Student-t cho chênh lệch trung bình ghép cặp.
    Trả về (mean_diff, low, high). Nếu khoảng không chứa 0 => chênh lệch có ý nghĩa.
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    mean = diff.mean()
    se = stats.sem(diff)                       # sai số chuẩn của trung bình
    df = len(diff) - 1
    low, high = stats.t.interval(confidence, df, loc=mean, scale=se)
    return mean, low, high


def ci_diff_bootstrap(scores_a, scores_b, confidence=0.95, n_boot=10000, random_state=42):
    """CI kiểu percentile bootstrap cho chênh lệch trung bình ghép cặp.
    Không giả định phân phối; với ít fold (5) thì kém ổn định hơn ci_diff_t.
    Trả về (mean_diff, low, high).
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    rng = np.random.default_rng(random_state)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(n_boot)])
    alpha = (1 - confidence) / 2
    return diff.mean(), np.quantile(boot, alpha), np.quantile(boot, 1 - alpha)


#! Hiệu chỉnh so sánh bội
def bonferroni_correction(pvalues, alpha=0.05):
    """Hiệu chỉnh Bonferroni cho m phép so sánh. Nhân p-value với m để bù việc
    so nhiều cặp cùng lúc. Trả về (reject: bool[], p_adjusted: float[]).
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    p_adj = np.minimum(p * m, 1.0)
    reject = p_adj < alpha
    return reject, p_adj


#! Kiểm tra calibration
def brier_multiclass(y_true, y_proba, labels=(0, 1, 2)):
    """Brier score đa lớp: trung bình bình phương sai lệch giữa xác suất dự đoán
    và nhãn one-hot. Càng nhỏ càng tốt. Bổ sung cho log loss.
    """
    onehot = np.eye(len(labels))[np.asarray(y_true)]
    return np.mean(np.sum((np.asarray(y_proba) - onehot) ** 2, axis=1))


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """Expected Calibration Error (ECE) theo độ tự tin. Chia dự đoán thành n_bins
    thùng theo xác suất lớp dự đoán; trong mỗi thùng so độ tự tin trung bình với
    tỉ lệ đúng thực tế. Trả về số trong [0, 1], càng nhỏ = xác suất càng đáng tin.
    """
    y_proba = np.asarray(y_proba)
    y_true = np.asarray(y_true)
    conf = y_proba.max(axis=1)                 # độ tự tin = xác suất lớp dự đoán
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return ece


def reliability_data(y_true, y_proba, class_idx, n_bins=10):
    """Dữ liệu vẽ reliability diagram cho một lớp (kiểu one-vs-rest).
    Trả về (prob_true, prob_pred): trục y = tần suất thực tế, trục x = xác suất
    dự đoán trung bình mỗi bin. Đường chéo 45 độ = hiệu chỉnh hoàn hảo.
    """
    y_bin = (np.asarray(y_true) == class_idx).astype(int)
    prob_true, prob_pred = calibration_curve(y_bin, np.asarray(y_proba)[:, class_idx],
                                             n_bins=n_bins)
    return prob_true, prob_pred
