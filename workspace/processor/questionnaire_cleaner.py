# survey_data_processor.py

import pandas as pd
import re
from typing import List, Dict, Any

class QuestionnaireCleaner:
    """
    问卷数据清洗处理器（Processor）
    输入：原始 DataFrame（含中文列名）
    输出：标准化清洗后的 DataFrame（符合 Codebook 规范）
    """

    # 配置：部门映射
    DEPT_MAPPING = {
        "生产": 1,
        "研发": 2,
        "销售": 3,
        "职能": 4,
        "其他": 5
    }

    # 配置：福利选项列表（用于生成虚拟变量）
    BENEFIT_OPTIONS = ["五险一金", "带薪年假", "补充医疗"]

    def __init__(self):
        self.raw_df: pd.DataFrame = None
        self.cleaned_df: pd.DataFrame = None

    def process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """主入口：执行完整清洗流程"""
        self.raw_df = raw_df.copy()
        df = self.raw_df.copy()

        # 阶段1：元数据标准化
        df = self._standardize_datetime(df)
        df = self._extract_numeric_fields(df)

        # 阶段2：结构化清洗
        df = self._encode_department(df)
        df = self._handle_satisfaction_and_workload(df)
        df = self._process_benefits(df)
        df = self._extract_other_specify(df)

        # 阶段3：缺失值与异常处理
        df = self._handle_missing_values(df)
        df = self._flag_anomalies(df)  # 可选：标记异常（此处仅保留）

        self.cleaned_df = df
        return self.cleaned_df

    def _standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一提交时间格式"""
        df["submit_time"] = pd.to_datetime(
            df["提交时间"], 
            format="mixed",  # 自动推断格式
            errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df

    def _extract_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """从含单位的字符串中提取数值"""
        def extract_number(series: pd.Series) -> pd.Series:
            # 移除非数字字符（保留小数点和负号）
            cleaned = series.astype(str).str.replace(r"[^\d.-]", "", regex=True)
            # 转为数值，无法转换的设为 NaN
            return pd.to_numeric(cleaned, errors="coerce")

        df["age"] = extract_number(df["年龄"])
        df["tenure_years"] = extract_number(df["工作年限"])
        return df

    def _encode_department(self, df: pd.DataFrame) -> pd.DataFrame:
        """部门文本 → 数值编码"""
        df["department"] = df["所属部门"].map(self.DEPT_MAPPING)
        return df

    def _handle_satisfaction_and_workload(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理满意度（正向）与工作负荷（反向计分）"""
        # 满意度：直接转数值
        df["satisfaction"] = pd.to_numeric(df["满意度"], errors="coerce")

        # 工作负荷：反向计分（6 - 原值），先转数值
        workload_raw = pd.to_numeric(df["工作负荷"], errors="coerce")
        df["workload"] = 6 - workload_raw  # 反向计分
        return df

    def _process_benefits(self, df: pd.DataFrame) -> pd.DataFrame:
        """多选题拆分为虚拟变量"""
        # 保留原始字段（供追溯）
        df["benefits_raw"] = df["福利选项"].fillna("")

        # 初始化虚拟变量列
        for benefit in self.BENEFIT_OPTIONS:
            col_name = f"benefit_{self._to_snake_case(benefit)}"
            df[col_name] = df["benefits_raw"].str.contains(benefit, na=False).astype(int)

        return df

    def _extract_other_specify(self, df: pd.DataFrame) -> pd.DataFrame:
        """从“其他说明”中提取 〖...〗 内容"""
        pattern = r"〖(.*?)〗"
        df["other_dept_specify"] = (
            df["其他说明"]
            .astype(str)
            .apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else "")
        )
        # 若部门不是“其他”，则清空说明
        df.loc[df["department"] != 5, "other_dept_specify"] = ""
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """将 NaN 替换为 -99（未回答）"""
        numeric_cols = ["age", "tenure_years", "satisfaction", "workload"]
        df[numeric_cols] = df[numeric_cols].fillna(-99)

        # 福利原始字段：空值转为空字符串（已在 _process_benefits 中处理）
        return df

    def _flag_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        （可选）标记异常值，此处仅保留原始值，不修改
        实际项目中可添加日志记录或单独标记列
        """
        # 示例：年龄 < 18 或 > 70
        # df["age_anomaly"] = (df["age"] < 18) | (df["age"] > 70)
        return df

    @staticmethod
    def _to_snake_case(chinese: str) -> str:
        """简单中文转下划线命名（实际可用拼音或预定义映射）"""
        mapping = {
            "五险一金": "insurance",
            "带薪年假": "vacation",
            "补充医疗": "medical"
        }
        return mapping.get(chinese, "unknown")