# pdf_context.py

PDF_HELP_TEXTS = {
    0: { # Page 1: Introduction and Framework
        0: """Universities are currently struggling to balance strict budget constraints and technical efficiency with the societal expectations of the United Nations' 17 Sustainable Development Goals. Sustainability has shifted from a minor concern to a major standard by which institutions are judged.""",
        1: """Previous research indicates a major disconnect between mathematical efficiency evaluations and qualitative sustainability assessments. Traditional models reward institutions for simply 'doing more with less,' which frequently conflicts with long-term sustainability objectives.""",
        2: """This paper introduces an exploratory analysis comparing efficiency data from the Sapientia Observatory with sustainability scores from THE Impact and UI GreenMetric rankings. The authors use correlation analysis to identify 'cross-drivers' that show how efficiency and sustainability influence one another.""",
        3: """By analyzing these cross-drivers, the researchers divide universities into four specific clusters based on their efficiency and sustainability performance. This quantitative clustering is then used to guide qualitative research into the actual reporting and practices of Italian universities.""",
        "fallback": "This section introduces the core challenge of balancing financial efficiency with sustainability goals, and outlines the study's methodological approach to creating institutional profiles."
    },
    1: { # Page 2: Literature Review and Rankings Context
        0: """A review of 435 articles shows that academic research relies heavily on measurable inputs and outputs while ignoring the qualitative behaviors that actually drive institutional change. Current university evaluation models are disjointed, leading to an incomplete picture where sustainability is often treated as just a symbolic gesture.""",
        1: """The United Nations' Sustainable Development Goals (SDGs) act as the primary framework for global sustainability in this study. The literature emphasizes that universities must directly integrate these goals into their core operations to create societal change.""",
        2: """The UI GreenMetric measures the physical and operational infrastructure of a 'green campus'. In contrast, the THE Impact Rankings measure a university's broader societal and economic impacts across the 17 SDGs.""",
        3: """Sustainability reports are essential tools for institutional transparency and act as catalysts for organizational change. However, while Italian universities adhere to reporting standards and have strategic plans, they struggle to adopt dedicated Sustainability Plans.""",
        4: """To create a fair comparative analysis, the data collected for this study was standardized using the year 2023 as a baseline.""",
        "fallback": "This page reviews existing literature on sustainability in higher education, highlighting the gap between quantitative rankings and the actual implementation of the UN's 2030 Agenda."
    },
    2: { # Page 3: Efficiency Data Collection
        0: """The efficiency data for the study was extracted from the Sapientia Observatory database. This specific dataset tracks the financial resource inputs and academic outputs of Italian universities.""",
        1: """While the initial database contained 60 institutions, the study required universities to have data present across all three sources (Sapientia, THE Impact, and UI GreenMetric). This necessary filtering reduced the final dataset to a sample of 29 specific Italian universities.""",
        "fallback": "This section explains the data collection process for efficiency metrics, specifically detailing how the sample size was narrowed down to 29 Italian universities."
    },
    3: { # Page 4: Sustainability Data and Web Scraping
        0: """Sustainability performance data was gathered from two distinct ranking systems to capture both operational metrics and societal impact.""",
        1: """The THE Impact Rankings utilize the structure of the UN's SDGs. They translate the overarching goals into distinct, weighted quantitative indicators to measure a university's performance.""",
        2: """The UI GreenMetric utilizes its own unique methodology tailored specifically to assess environmental commitment.""",
        3: """To supplement the quantitative ranking data, the researchers built an automated web-scraping tool to collect public sustainability information from university websites. This scraper was designed as an exploratory tool to identify reporting intensity and institutional transparency, rather than to create a formal ranking.""",
        "fallback": "This page details the sustainability frameworks used (THE Impact and UI GreenMetric) and introduces an automated web-scraping tool built to evaluate institutional reporting visibility."
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
        return "This section discusses the broader findings of aligning university operational efficiency with qualitative sustainability reporting."
 
    return page_data.get(para_idx, page_data.get("fallback"))