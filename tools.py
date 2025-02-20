from langchain_core.tools import tool


@tool
def reClean() -> str:
    """Re-Clean Data"""
    print("1.1 Data is being recleaned")
    return "Data has been recleaned and formatted successfully."

dataTools = [reClean]


@tool
def statTests() -> str:
    """Run Pandas tests"""
    print("2.1 Data is being tested using pandas")
    return "Data has been tested with the p-test"

@tool
def visualization() -> str:
    """Genrate matplot visualizations"""
    print("2.2 Data is displayed using matplot")
    return "Graph has been displayed uasing matplot"

analysisTools = [statTests, visualization]



@tool
def summarize() -> str:
    """Summarize findings based on the analysis"""
    print("3.1 Findings are being summarized")
    return "It seems like there's a strong correlationg with groundwater pollution and constructing on sand"

@tool
def reportFindings() -> str:
    """Report findings in detail based on the analysis"""
    print("3.2 All findings")
    return "Studies and observations suggest that there is a significant relationship between groundwater pollution and construction on sandy soil. This correlation arises because sandy soil has high permeability, meaning that water and other substances can pass through it more easily compared to other soil types. When buildings, industrial facilities, or other structures are constructed on sandy terrain, pollutants from construction materials, chemicals, or waste can seep into the ground more rapidly. This increases the risk of contaminants reaching and polluting the groundwater supply. Additionally, because sand does not provide the same level of natural filtration as denser soils like clay, harmful substances are less likely to be trapped before they reach the groundwater."

reportTools = [summarize, reportFindings]



allTools = [reClean, statTests, visualization, summarize, reportFindings]