# pdf_context.py

# Dictionary structure: { page_number: { paragraph_index: "Help Text", "fallback": "Page Summary" } }
PDF_HELP_TEXTS = {
    0: { # Page 1: Introduction
        1: "This paragraph introduces the core problem: universities are pressured to balance traditional operational efficiency with the UN's 17 Sustainable Development Goals (SDGs)[cite: 1].",
        2: "Here, the authors highlight a 'methodological schism'[cite: 1]. Traditional efficiency models (like DEA) reward 'doing more with less,' which often conflicts with long-term sustainability goals[cite: 1].",
        3: "This outlines the paper's method: using simple correlation analysis to compare efficiency data (Sapientia) with sustainability rankings (THE Impact and UI GreenMetric) to find 'cross-drivers'[cite: 1].",
        4: "The authors explain they will group universities into four clusters (e.g., High Efficiency-High Sustainability) based on their scores, which will guide the qualitative analysis[cite: 1].",
        "fallback": "This introduction outlines the goal to bridge quantitative efficiency metrics with qualitative sustainability rankings in Italian universities[cite: 1]."
    },
    1: { # Page 2: Literature Review
        0: "Based on a review of 435 articles, this paragraph shows that current research focuses too much on measurable inputs/outputs and ignores the qualitative drivers of change[cite: 1].",
        1: "The review concludes that because efficiency and sustainability are measured separately, current rankings give an incomplete picture, treating sustainability merely as a symbolic gesture[cite: 1].",
        2: "This section introduces the UN 2030 Agenda and its 17 Sustainable Development Goals (SDGs) as the primary framework, noting institutions must move beyond symbolic adoption[cite: 1].",
        3: "Here, two main rankings are compared: UI GreenMetric focuses on campus environmental operations, while THE Impact Rankings focus on broader societal impacts[cite: 1].",
        4: "This paragraph explains that sustainability reports ('Bilancio di Sostenibilità') are crucial tools for institutional accountability and organizational change[cite: 1].",
        5: "In Italy, while universities use standard reporting frameworks, there is still a significant gap in actually adopting dedicated Sustainability Plans[cite: 1].",
        "fallback": "This page reviews existing literature, noting a disconnect between mathematical efficiency models and actual sustainability integration[cite: 1]."
    },
    2: { # Page 3: Data Collection
        0: "This paragraph explains that the dataset was standardized using 2023 as the baseline year to ensure robust comparative analysis[cite: 1].",
        1: "The efficiency metrics were sourced from the Sapientia Observatory, which tracks resource inputs and academic outputs[cite: 1].",
        2: "The dataset was restricted to 29 Italian universities that appeared across all three data sources (Sapientia, UI GreenMetric, and THE Impact Rankings)[cite: 1].",
        "fallback": "This page details the data collection, specifically focusing on the 29 Italian universities filtered from the Sapientia Observatory[cite: 1]."
    },
    3: { # Page 4: THE Impact Data
        0: "This notes that the THE Impact Rankings use the UN's SDGs as a structural framework, translating them into specific quantifiable institutional indicators[cite: 1].",
        1: "This references Table 1, which outlines how the THE framework measures specific goals (like poverty, health, and gender equality)[cite: 1].",
        "fallback": "This page breaks down how the THE Impact Rankings translate UN SDGs into measurable data points[cite: 1]."
    },
    4: { # Page 5: GreenMetric & Scraper
        0: "This notes the UI GreenMetric uses a different methodology designed specifically to measure environmental commitment[cite: 1].",
        1: "To gather more context, the researchers built an automated Python web-scraper to collect public sustainability information from university websites[cite: 1].",
        2: "The scraper used a recursive crawling architecture, starting from manual root pages since sustainability info is often scattered across different subdomains[cite: 1].",
        3: "The crawler was restricted by depth and keyword filters (like 'sustainability', 'SDG', 'climate') to keep the search focused[cite: 1].",
        4: "It extracted HTML text and used PyMuPDF to extract text from linked PDF documents, like strategic plans and governance reports[cite: 1].",
        "fallback": "This page explains the UI GreenMetric framework and details the custom Python web-scraper built to analyze university websites[cite: 1]."
    },
    6: { # Page 7: Quantitative Analysis
        0: "This section explores how internal efficiency intersects with multidimensional sustainability metrics to identify core drivers[cite: 1].",
        1: "To fix missing data (NaNs) in the THE rankings, they only used metrics with at least 5 entries and calculated a 'stability score' (mean divided by standard deviation)[cite: 1].",
        2: "The analysis maps top positive correlations, showing how highly optimized resource management aligns with sustainability[cite: 1].",
        4: "It highlights that financially robust universities with strong research profiles naturally support sustainable campus integration (SDG 11 and 12)[cite: 1].",
        5: "Conversely, strong negative correlations reveal institutional trade-offs and operational barriers[cite: 1].",
        "fallback": "This page analyzes the quantitative data, finding that strong finances often support campus sustainability, while exposing other trade-offs[cite: 1]."
    },
    7: { # Page 8: Negative Correlates & Clusters
        0: "A major negative finding: intense focus on elite research (like publishing in Nature/Science) and high spending per student severely conflicts with climate action and educational equality goals[cite: 1].",
        1: "This section uses cross-correlation to find specific factors (drivers) that help or hurt sustainability performance[cite: 1].",
        2: "The top positive driver is the prevalence of Postgraduate First-Level Master's Degrees[cite: 1].",
        4: "A negative driver is an older average age of staff, suggesting traditional mindsets or older infrastructure act as barriers to sustainable transitions[cite: 1].",
        "fallback": "This page highlights negative trade-offs, showing how an over-focus on elite research rankings can hinder broader sustainability goals[cite: 1]."
    },
    8: { # Page 9: The Quadrants
        0: "The data allows universities to be mapped onto a 4-quadrant matrix based on Efficiency and Sustainability scores[cite: 1].",
        1: "Top-Right (Green): High Efficiency / High Sustainability. These institutions lead in both operational and ESG metrics[cite: 1].",
        2: "Bottom-Right (Red): High Efficiency / Low Sustainability. Highly productive universities that haven't translated that success into sustainability rankings[cite: 1].",
        3: "Top-Left (Blue): Low Efficiency / High Sustainability. Institutions that prioritize sustainability despite lower technical efficiency[cite: 1].",
        4: "Bottom-Left (Grey): Low Efficiency / Low Sustainability. Institutions facing structural challenges in both areas[cite: 1].",
        "fallback": "This page defines a diagnostic 4-quadrant matrix to categorize universities based on their Efficiency and Sustainability profiles[cite: 1]."
    }
}

def get_help_text_for_paragraph(page_num, para_idx):
    """
    Retrieves the specific help text for a given page and paragraph index.
    Falls back to a general page summary if the specific paragraph isn't mapped.
    Falls back to a generic document summary if the page isn't mapped.
    """
    page_data = PDF_HELP_TEXTS.get(page_num)
    
    if not page_data:
        return "This section discusses the broader findings of aligning university operational efficiency with qualitative sustainability reporting[cite: 1]."
        
    # Attempt to get the exact paragraph, otherwise use the page's fallback
    return page_data.get(para_idx, page_data.get("fallback"))