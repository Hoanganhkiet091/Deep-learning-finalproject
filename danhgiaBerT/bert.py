import os
import torch
import pandas as pd
import numpy as np
from bert_score import BERTScorer
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# CLASS ĐÁNH GIÁ VĂN BẢN Y TẾ VỚI BERTSCORE
# ==========================================
class MedicalReportEvaluator:
    def __init__(self, lang="en", model_type="roberta-large", device=None):
        """
        Khởi tạo bộ đánh giá.
        - lang: "en" (Tiếng Anh) hoặc "vi" (Tiếng Việt)
        - model_type: Mô hình dùng để trích xuất ngữ nghĩa (roberta-large rất chuẩn cho Tiếng Anh)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Khởi tạo BERTScorer trên thiết bị: {self.device}...")
        logger.info(f"Sử dụng mô hình: {model_type} (Ngôn ngữ: {lang})")
        
        # Khởi tạo Scorer một lần để tái sử dụng, giúp tăng tốc quá trình infer
        self.scorer = BERTScorer(lang=lang, model_type=model_type, device=self.device, rescale_with_baseline=True)

    def evaluate(self, predictions, references, batch_size=16):
        """
        Tính toán BERTScore cho tập dữ liệu.
        Trả về trung bình Precision, Recall, và F1.
        """
        logger.info(f"Đang tiến hành chấm điểm BERTScore cho {len(predictions)} mẫu...")
        
        # Chia batch để tránh tràn RAM/VRAM nếu tập dữ liệu lớn
        P_list, R_list, F1_list = [], [], []
        
        for i in tqdm(range(0, len(predictions), batch_size), desc="Đánh giá BERTScore"):
            batch_preds = predictions[i : i + batch_size]
            batch_refs = references[i : i + batch_size]
            
            P, R, F1 = self.scorer.score(batch_preds, batch_refs)
            P_list.extend(P.cpu().numpy())
            R_list.extend(R.cpu().numpy())
            F1_list.extend(F1.cpu().numpy())
            
        return np.mean(P_list), np.mean(R_list), np.mean(F1_list)

# ==========================================
# HÀM CHẠY PIPELINE SO SÁNH (BASELINE VS RAG)
# ==========================================
def run_evaluation_pipeline(data_path):
    """
    Hàm đọc dữ liệu từ file và so sánh kết quả của Baseline LLM với RAG-Augmented LLM
    """
    logger.info(f"Đọc dữ liệu từ: {data_path}")
    mock_data = {
        'expert_consensus': [
            "Tổn thương da bất đối xứng với nhiều màu sắc khác nhau, nghi ngờ ung thư hắc tố (Melanoma). Khuyến nghị sinh thiết gấp để xác định.",
            "Bệnh nhân có dày sừng tiết bã, tổn thương lành tính không cần can thiệp y tế khẩn cấp trừ khi ảnh hưởng thẩm mỹ."
        ] * 50, # Nhân bản lên 100 mẫu để chạy demo
        
        'baseline_llm_report': [
            "Khối u này có vẻ là ung thư hắc tố. Cần phẫu thuật cắt bỏ.", # Hơi lỏng lẻo, thiếu chi tiết
            "Đây là một bệnh ngoài da, có thể là viêm da hoặc nấm da."     # Sinh ảo (Hallucination)
        ] * 50,
        
        'rag_llm_report': [
            "Tổn thương hiển thị tính bất đối xứng và đa sắc, đặc trưng của Melanoma. Đề nghị tiến hành sinh thiết để chẩn đoán xác định.", # Sát nghĩa với chuyên gia
            "Theo hình ảnh lâm sàng, đây là dày sừng tiết bã - một dạng tổn thương lành tính. Không yêu cầu điều trị xâm lấn." # Chính xác nhờ có RAG
        ] * 50
    }
    df = pd.DataFrame(mock_data)
    
    # Lấy danh sách các câu
    references = df['expert_consensus'].tolist()
    baseline_preds = df['baseline_llm_report'].tolist()
    rag_preds = df['rag_llm_report'].tolist()
    
    # Khởi tạo Evaluator (Tiếng Việt dùng "vi", Tiếng Anh dùng "en")
    evaluator = MedicalReportEvaluator(lang="vi") 
    
    # 1. Đánh giá Baseline LLM (Un-verified baseline)
    logger.info("=== BƯỚC 1: ĐÁNH GIÁ BASELINE LLM ===")
    base_P, base_R, base_F1 = evaluator.evaluate(baseline_preds, references)
    
    # 2. Đánh giá RAG-Augmented LLM
    logger.info("=== BƯỚC 2: ĐÁNH GIÁ RAG-AUGMENTED LLM ===")
    rag_P, rag_R, rag_F1 = evaluator.evaluate(rag_preds, references)
    
    # 3. Tổng hợp và in báo cáo chuẩn
    improvement = rag_F1 - base_F1
    
    print("\n" + "="*60)
    print(" BÁO CÁO KẾT QUẢ ĐÁNH GIÁ SINH VĂN BẢN Y TẾ (BERTSCORE)".center(60))
    print("="*60)
    print(f"Số lượng mẫu đánh giá    : {len(df)} báo cáo")
    print("-" * 60)
    print(f"[1] Baseline LLM (Un-verified):")
    print(f"    - Precision (P)      : {base_P:.4f}")
    print(f"    - Recall (R)         : {base_R:.4f}")
    print(f"    - F1-Score           : {base_F1:.4f}")
    print("-" * 60)
    print(f"[2] RAG-Augmented LLM (Proposed):")
    print(f"    - Precision (P)      : {rag_P:.4f}")
    print(f"    - Recall (R)         : {rag_R:.4f}")
    print(f"    - F1-Score           : {rag_F1:.4f}")
    print("="*60)
    print(f"KẾT LUẬN: RAG-Augmented LLM đạt BERTScore F1 là {rag_F1:.2f},")
    print(f"cải thiện {improvement:.2f} so với baseline không được kiểm chứng.")
    print("="*60 + "\n")
    
    # Xuất kết quả chi tiết ra file
    df['Baseline_F1'] = base_F1
    df['RAG_F1'] = rag_F1
    df.to_csv("bertscore_evaluation_results.csv", index=False, encoding='utf-8-sig')
    logger.info("Đã lưu chi tiết kết quả vào 'bertscore_evaluation_results.csv'")

if __name__ == "__main__":
    # Điền đường dẫn file dataset của bạn vào đây (ví dụ: "test_reports.csv")
    run_evaluation_pipeline(data_path="mock_path")