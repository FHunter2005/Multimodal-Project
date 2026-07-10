
PDF_HELP_TEXTS = {
    0: { # Page 1: Introduction
        1: """This paragraph introduces the core conflict of the paper: universities are under massive pressure to maintain traditional operational efficiency—often measured by financial savings and research throughput—while simultaneously trying to adopt the UN's 17 Sustainable Development Goals (SDGs). The authors argue that these two missions often compete for the same limited resources.""",
        2: """The authors identify a 'methodological schism' here. Traditional models like Data Envelopment Analysis (DEA) prioritize 'doing more with less' (efficiency), but sustainability often requires 'doing things differently' (quality). By focusing only on efficiency, universities often overlook the hidden costs to sustainability.""",
        3: """This section outlines the paper's methodological approach: a correlation analysis. Rather than reinventing the wheel, the authors compare established efficiency metrics (Sapientia) against sustainability performance rankings (THE Impact and UI GreenMetric). The goal is to find 'cross-drivers'—factors that improve both operational productivity and sustainability simultaneously.""",
        4: """This explains the logic behind the authors' clustering strategy. By grouping universities into four distinct quadrants (e.g., High Efficiency-High Sustainability), they shift the analysis from abstract data to actionable policy profiles. This allows them to see which institutional characteristics actually move the needle for sustainability.""",
        "fallback": "This introduction outlines the central tension: the struggle to bridge quantitative efficiency metrics with qualitative sustainability rankings in Italian universities."
    },
    1: { # Page 2: Literature Review
        0: """Based on a meta-analysis of 435 articles, this section reveals a major gap in academia: current research is obsessed with measurable inputs (like budget) and outputs (like citations), but largely ignores the qualitative, behavioral drivers of institutional change. It suggests that most existing papers miss the 'human element' of sustainability.""",
        1: """The literature review concludes that efficiency and sustainability are currently siloed—they are measured by different agencies using different criteria. The paper argues that current rankings provide an incomplete, almost superficial picture, where sustainability is often treated as a PR 'check-box' exercise rather than an institutional priority.""",
        2: """This introduces the UN 2030 Agenda (SDGs) as the normative framework. The authors emphasize that institutional adoption of these goals is often purely symbolic. The critical challenge isn't just signing onto the agenda, but integrating it into daily research and teaching practices.""",
        3: """This section distinguishes between the two ranking frameworks: UI GreenMetric focuses on the 'hardware' of sustainability (campus energy, waste management, transportation), while THE Impact Rankings look at the 'software' (societal impact, research, gender equality). It’s an important distinction for university leadership.""",
        4: """This highlights the 'Bilancio di Sostenibilità' (Sustainability Report) as the primary tool for accountability. However, the authors note that writing the report is not the same as implementing the strategy—reporting is a diagnostic tool, not the cure itself.""",
        5: """The review ends on a sobering note for the Italian context: despite having robust reporting frameworks, Italian universities lag behind in actually adopting dedicated Sustainability Plans that dictate long-term budget and policy shifts.""",
        "fallback": "This page reviews existing literature, noting a deep disconnect between mathematical efficiency models and actual sustainability integration in Italian higher education."
    },
    6: { # Page 7: Quantitative Analysis
        0: """This section explores the 'cross-over' points where internal resource efficiency aligns with, or clashes against, multidimensional sustainability metrics. The goal is to identify core 'drivers'—specific university attributes that reliably predict sustainable success across different frameworks.""",
        1: """To manage the messy reality of data, the authors had to perform 'data cleaning.' They handled missing values (NaNs) by filtering metrics with fewer than 5 entries and calculating a 'stability score'—essentially measuring if a university’s performance is consistent or just a lucky outlier.""",
        2: """The analysis maps top positive correlations, suggesting that the most optimized, well-managed institutions have a 'virtuous cycle.' When a university is efficient with resource management, it often manages its energy, waste, and campus life better, too.""",
        4: """This paragraph finds that financially robust universities with strong research profiles have a natural advantage. These institutions have the capital to invest in 'greening' their infrastructure, which makes hitting SDG 11 (Sustainable Cities) and SDG 12 (Responsible Consumption) much easier.""",
        5: """Conversely, the negative correlations identify the 'structural barriers.' These aren't just administrative failures; they are indicative of institutional trade-offs—sometimes, optimizing for one ranking framework inherently degrades performance in another.""",
        "fallback": "This analysis identifies that while strong finances often support campus sustainability, there are specific, measurable trade-offs in operational management."
    },
    7: { # Page 8: Negative Correlates & Clusters
        0: """This is one of the most critical findings: elite research intensity (e.g., publishing in top-tier journals) and high per-student spending often show an inverse relationship with climate action and educational equality. This suggests that the 'Publish or Perish' culture may inadvertently come at the cost of broader institutional sustainability.""",
        1: """The authors use cross-correlation here to move beyond simple 'good/bad' labels and isolate the specific variables that drive sustainability up or down, helping universities diagnose their unique problems.""",
        2: """Interestingly, the strongest positive driver for sustainability is the prevalence of 'Postgraduate First-Level Master's Degrees.' This suggests that universities with strong vocational and advanced training programs are more effectively integrating sustainability into their curriculum.""",
        4: """The authors point to the 'older average age of staff' as a significant negative driver. This isn't just about age—it represents institutional inertia, where legacy mindsets and older, less efficient physical infrastructure create a 'lock-in' effect that makes sustainable transition harder.""",
        "fallback": "This page highlights difficult trade-offs: an extreme obsession with elite research rankings can unintentionally hinder progress on broader sustainability goals."
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
 
    return page_data.get(para_idx, page_data.get("fallback"))