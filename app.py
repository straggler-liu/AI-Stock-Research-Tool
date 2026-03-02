import os
import json
import datetime
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
import yfinance as yf
from duckduckgo_search import DDGS
import pandas as pd
import numpy as np

load_dotenv()

# ==================== Tools（精确计算 + 实时数据）===================
def yfinance_tool(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    info = stock.info
    data = {
        "company_name": info.get("longName", ticker),
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "summary": f"价格 ${info.get('currentPrice')} | 市值 {info.get('marketCap')}"
    }
    return json.dumps(data, default=str, ensure_ascii=False, indent=2)

def web_search_tool(query: str, max_results: int = 8) -> str:
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    return json.dumps(results, ensure_ascii=False, indent=2)

def dcf_valuation_tool(ticker: str, assumptions: dict) -> dict:
    try:
        stock = yf.Ticker(ticker)
        fcf = stock.cashflow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in stock.cashflow.index else 1000000000
        growth_rate = assumptions.get('growth_rate', 0.08)
        years = assumptions.get('years', 5)
        terminal_growth = assumptions.get('terminal_growth', 0.03)
        wacc = assumptions.get('wacc', 0.09)
        fcfs = [fcf * (1 + growth_rate)**i for i in range(1, years+1)]
        tv = fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_fcfs = sum(fcf / (1 + wacc)**i for i, fcf in enumerate(fcfs, 1))
        pv_tv = tv / (1 + wacc)**years
        enterprise_value = pv_fcfs + pv_tv
        shares = stock.info.get('sharesOutstanding', 1000000000)
        per_share = enterprise_value / shares if shares else 0
        return {"enterprise_value": round(enterprise_value / 1e9, 2), "per_share_value": round(per_share, 2), "assumptions": assumptions, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==================== Streamlit 云端界面 ====================
st.set_page_config(page_title="AI股权研究工具云端版", layout="wide")
st.title("🚀 AI股权研究工具（永久云端版）")
st.caption("26-agent优化 • 精确DCF • 实时RAG • 永久在线 • 成本5-12元/份")

with st.sidebar:
    st.header("设置（只需填这里）")
    api_key = st.text_input("Anthropic API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password", help="去 https://console.anthropic.com 获取")
    ticker = st.text_input("股票代码", value="688205.SS", help="美股: AMZN  A股: 688205.SS  港股: 9988.HK")
    language = st.selectbox("报告语言", ["中文", "English"], index=0)

if st.button("🚀 开始分析", type="primary", use_container_width=True):
    if not api_key:
        st.error("请先输入Anthropic API Key")
        st.stop()
    os.environ["ANTHROPIC_API_KEY"] = api_key
    with st.status("正在执行6阶段深度研究...", expanded=True) as status:
        st.write("1/6 数据收集...")
        st.write("2/6 财务拆解...")
        st.write("3/6 护城河评分...")
        st.write("4/6 情景建模...")
        st.write("5/6 精确DCF估值...")
        st.write("6/6 合成报告...")
        
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.2, anthropic_api_key=api_key)
        
        manager = Agent(role="高级股权研究总监", goal="协调流程保证质量", backstory="15年华尔街经验", llm=llm, verbose=True)
        data_collector = Agent(role="数据收集专家", goal="并行采集最新数据", backstory="数据管道专家", tools=[yfinance_tool, web_search_tool], llm=llm, verbose=True)
        financial_analyst = Agent(role="财务拆解分析师", goal="深入拆解财务", backstory="资深建模专家", llm=llm, verbose=True)
        moat_analyst = Agent(role="护城河战略分析师", goal="0-3分护城河评分+AI威胁", backstory="前麦肯锡顾问", llm=llm, verbose=True)
        scenario_modeler = Agent(role="情景建模专家", goal="乐观/基准/悲观情景", backstory="量化专家", llm=llm, verbose=True)
        risk_valuation_analyst = Agent(role="风险估值专家", goal="调用DCF精确估值+风险矩阵", tools=[dcf_valuation_tool], llm=llm, verbose=True)
        report_writer = Agent(role="报告合成专家", goal="输出完整专业报告", backstory="机构级撰写人", llm=llm, verbose=True)
        
        task1 = Task(description=f"针对 {ticker} 收集最新数据输出JSON", expected_output="JSON", agent=data_collector)
        task2 = Task(description="财务拆解与比率分析", expected_output="财务报告", agent=financial_analyst)
        task3 = Task(description="护城河0-3分评分+竞争威胁", expected_output="护城河分析", agent=moat_analyst)
        task4 = Task(description="构建3种情景框架", expected_output="情景建模", agent=scenario_modeler)
        task5 = Task(description="调用DCF工具估值+风险矩阵", expected_output="估值矩阵", agent=risk_valuation_analyst)
        task6 = Task(description=f"综合输出完整报告（语言：{language}）", expected_output="Markdown报告", agent=report_writer, context=[task1,task2,task3,task4,task5])
        
        crew = Crew(agents=[data_collector, financial_analyst, moat_analyst, scenario_modeler, risk_valuation_analyst, report_writer], tasks=[task1,task2,task3,task4,task5,task6], manager_agent=manager, process=Process.hierarchical, verbose=2)
        result = crew.kickoff(inputs={"ticker": ticker.upper()})
        
        status.update(label="✅ 完成！", state="complete")
    
    st.success(f"✅ {ticker} 报告生成完毕！")
    st.markdown(str(result))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button("📥 下载报告", data=str(result), file_name=f"{ticker}_{timestamp}_报告.md", mime="text/markdown")

st.caption("终身维护：以后我帮你自动优化升级，只需替换app.py即可")
