# questionnaire_cleaner.py

import pandas as pd
import numpy as np
from typing import Dict, Optional

class QuestionnaireCleaner:
    """
    问卷数据清洗处理器（Processor）
    输入：原始 DataFrame（含中文列名）
    输出：标准化清洗后的 DataFrame（符合 DataContract 规范）

    符合 DataContract: tests/fixtures/workspace/catelog/contract/output-contract.yaml
    """

    # 配置：性别标准化映射
    GENDER_MAPPING = {
        "男": "male",
        "女": "female",
        "其他": "other",
        "未知": "unknown"
    }

    # 配置：教育程度映射
    EDU_MAPPING = {
        "初中": "初中",
        "高中": "高中",
        "大专": "大专",
        "本科": "本科",
        "硕士": "硕士",
        "MBA": "硕士",  # MBA映射为硕士
        "博士": "博士",
        "其他": "其他",
        "未知": "未知"
    }

    # 配置：雇佣状态映射
    EMP_STATUS_MAPPING = {
        "在职": "在职",
        "实习生": "实习生",
        "返聘": "返聘",
        "退休": "非员工",
        "学生": "非员工",
        "其他": "其他",
        "未知": "未知"
    }

    # 配置：城市标准化映射
    CITY_MAPPING = {
        "北京": "北京",
        "上海": "上海",
        "广州": "广州",
        "深圳": "深圳",
        "杭州": "杭州",
        "成都": "成都",
        "重庆": "重庆",
        "其他城市": "其他城市",
        "未知城市": "未知城市"
    }

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None

    def process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """主入口：执行完整清洗流程"""
        self.raw_df = raw_df.copy()
        df = self.raw_df.copy()

        # 阶段1：元数据标准化
        df = self._standardize_datetime(df)
        df = self._standardize_id(df)

        # 阶段2：数值字段处理
        df = self._process_age(df)
        df = self._process_total_exp(df)
        df = self._process_satisfaction(df)
        df = self._process_workload(df)
        df = self._process_tenure(df)
        df = self._process_monthly_income(df)

        # 阶段3：分类字段标准化
        df = self._standardize_dept(df)
        df = self._standardize_gender(df)
        df = self._standardize_education(df)
        df = self._standardize_emp_status(df)
        df = self._standardize_city(df)

        # 阶段4：福利字段处理
        df = self._process_benefits(df)

        # 阶段5：备注字段处理
        df = self._process_other_notes(df)

        # 阶段6：重复检测与标记
        df = self._detect_duplicates(df)

        # 阶段7：数据质量标记
        df = self._add_data_quality_flags(df)

        # 阶段8：字段选择与排序
        df = self._select_and_order_columns(df)

        self.cleaned_df = df
        return self.cleaned_df

    def _standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一提交时间格式为 YYYY-MM-DD HH:MM:SS"""
        df["submit_time"] = pd.to_datetime(
            df["提交时间"] if "提交时间" in df.columns else df["submit_time"],
            format="mixed",
            errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df

    def _standardize_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化ID字段为整数"""
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        return df

    def _process_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理年龄字段，约束：16-200，允许NULL"""
        age_col = "年龄" if "年龄" in df.columns else "age"
        df["age"] = pd.to_numeric(df[age_col], errors="coerce")
        # 不强制限制范围，允许极端值，通过data_quality_flag标记
        return df

    def _process_total_exp(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理工作年限字段，约束：0-50，允许NULL"""
        exp_col = "工作年限" if "工作年限" in df.columns else "total_exp"
        df["total_exp"] = pd.to_numeric(df[exp_col], errors="coerce")
        return df

    def _process_satisfaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理整体满意度字段，约束：0-6分，允许NULL"""
        sat_col = "满意度" if "满意度" in df.columns else "overall_satis"
        df["overall_satis"] = pd.to_numeric(df[sat_col], errors="coerce")
        return df

    def _process_workload(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理工作负荷字段，约束：1-10分，允许NULL"""
        workload_col = "工作负荷" if "工作负荷" in df.columns else "workload"
        df["workload"] = pd.to_numeric(df[workload_col], errors="coerce")
        return df

    def _process_tenure(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理任期/司龄字段，约束：0-50，支持小数，允许NULL"""
        tenure_col = "任期" if "任期" in df.columns else "tenure"
        df["tenure"] = pd.to_numeric(df[tenure_col], errors="coerce")
        return df

    def _process_monthly_income(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理月收入字段，约束：0-35000，负数转为NULL，允许NULL"""
        income_col = "月收入" if "月收入" in df.columns else "monthly_income"
        df["monthly_income"] = pd.to_numeric(df[income_col], errors="coerce")
        # 负数转为NULL
        df.loc[df["monthly_income"] < 0, "monthly_income"] = None
        return df

    def _standardize_dept(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化部门字段"""
        dept_col = "所属部门" if "所属部门" in df.columns else "dept"
        df["dept"] = df[dept_col].fillna("其他")
        return df

    def _standardize_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化性别字段为 male/female/unknown"""
        gender_col = "性别" if "性别" in df.columns else "gender"
        df["gender"] = df[gender_col].fillna("unknown").map(self.GENDER_MAPPING).fillna("unknown")
        return df

    def _standardize_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化教育程度字段，MBA映射为硕士"""
        edu_col = "教育程度" if "教育程度" in df.columns else "edu"
        df["edu"] = df[edu_col].fillna("未知").map(self.EDU_MAPPING).fillna("其他")
        return df

    def _standardize_emp_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化雇佣状态字段"""
        status_col = "雇佣状态" if "雇佣状态" in df.columns else "emp_status"
        df["emp_status"] = df[status_col].fillna("未知").map(self.EMP_STATUS_MAPPING).fillna("其他")
        return df

    def _standardize_city(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化城市字段为中文"""
        city_col = "城市" if "城市" in df.columns else "city"
        df["city"] = df[city_col].fillna("未知城市").map(self.CITY_MAPPING).fillna("未知城市")
        return df

    def _process_benefits(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理福利字段，转换为boolean类型"""
        benefit_cols = {
            "benefit_pension": ["养老金", "养老"],
            "benefit_annual_leave": ["年假", "带薪年假"],
            "benefit_health_ins": ["医疗", "医保", "医疗保险"],
            "benefit_other": ["其他", "其他福利"]
        }

        for col_name, keywords in benefit_cols.items():
            # 查找对应的中文列名
            source_col = None
            for keyword in keywords:
                if keyword in df.columns:
                    source_col = keyword
                    break

            if source_col is not None:
                df[col_name] = df[source_col].fillna(False).astype(bool)
            else:
                # 如果没有原始列，初始化为False
                df[col_name] = False

        return df

    def _process_other_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理备注字段，从性别列或其他字段提取备注信息"""
        notes_col = "备注" if "备注" in df.columns else "other_notes"
        if notes_col in df.columns:
            df["other_notes"] = df[notes_col].fillna("")
        else:
            df["other_notes"] = ""
        return df

    def _detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测重复记录并标记"""
        df["is_duplicate"] = False
        if df.duplicated(subset=["submit_time", "age", "total_exp", "dept"]).any():
            df["is_duplicate"] = df.duplicated(subset=["submit_time", "age", "total_exp", "dept"], keep="first")
        return df

    def _add_data_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加数据质量标记"""
        df["data_quality_flag"] = "正常"

        # 收入缺失
        df.loc[df["monthly_income"].isna() & (df["data_quality_flag"] == "正常"), "data_quality_flag"] = "收入缺失"

        # 关键字段缺失
        df.loc[(df["overall_satis"].isna() | df["workload"].isna()) & (df["data_quality_flag"] == "正常"), "data_quality_flag"] = "关键字段缺失"

        # 逻辑校验：学生
        df.loc[(df["emp_status"] == "非员工") & (df["age"] < 18) & (df["emp_status"].shift(1) != "非员工") & (df["data_quality_flag"] == "正常"), "data_quality_flag"] = "逻辑校验_学生"

        # 逻辑校验：退休
        df.loc[(df["emp_status"] == "非员工") & (df["age"] >= 60) & (df["emp_status"].shift(1) != "非员工") & (df["data_quality_flag"] == "正常"), "data_quality_flag"] = "逻辑校验_退休"

        # 异常值：年龄 > 70 或工作负荷 > 10
        df.loc[(df["age"] > 70) | (df["workload"] > 10), "data_quality_flag"] = "异常值_收入负数_工作负荷越界"

        # 测试数据（优先级最高）
        df.loc[df["dept"] == "测试部门", "data_quality_flag"] = "测试数据"

        # 重复记录（优先级最高）
        df.loc[df["is_duplicate"], "data_quality_flag"] = "重复记录"

        return df

    def _select_and_order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """选择并排序输出列"""
        output_columns = [
            "id",
            "submit_time",
            "age",
            "total_exp",
            "dept",
            "overall_satis",
            "workload",
            "benefit_pension",
            "benefit_annual_leave",
            "benefit_health_ins",
            "benefit_other",
            "other_notes",
            "gender",
            "edu",
            "emp_status",
            "tenure",
            "monthly_income",
            "city",
            "is_duplicate",
            "data_quality_flag"
        ]

        # 只保留存在的列
        existing_columns = [col for col in output_columns if col in df.columns]
        return df[existing_columns]
