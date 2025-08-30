import json
from connect_LLM_sample_test import get_related_subspace, get_response, parse_response_select_group, read_vis_list, \
    parse_response_select_insight
from asyncio import run

# get header list
def read_vis_list_vegalite(file_path):
    global header_list_vegalite
    header_list_vegalite = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Header:"):
                header = eval(line.split(":")[1].strip())
                insights = []
                i += 1
                while i < len(lines) and not lines[i].startswith("Header:"):
                    if lines[i].startswith("Insight"):
                        insight_type = lines[i + 1].split(":")[1].strip()
                        insight_score = float(lines[i + 2].split(":")[1].strip())
                        insight_category = lines[i + 3].split(":")[1].strip()
                        insight_description = lines[i + 4].split(":")[1].strip()
                        data_str = lines[i + 5]
                        index = data_str.index('Vega-Lite Json: ')
                        insight_vegalite = data_str[index + len('Vega-Lite Json: '):]
                        insights.append(
                            {"Type": insight_type, "Score": insight_score, "Category": insight_category,
                             "Description": insight_description, "Vega-Lite": insight_vegalite})
                        i += 6
                    else:
                        i += 1
                header_list_vegalite.append({"Header": header, "Insights": insights})
            else:
                i += 1
    return header_list_vegalite


# insight list, fully info
def read_vis_list_into_insights(file_path):
    global insight_list
    insight_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Header:"):
                header = eval(line.split(":")[1].strip())
                insights = []
                i += 1
                while i < len(lines) and not lines[i].startswith("Header:"):
                    if lines[i].startswith("Insight"):
                        insight_type = lines[i + 1].split(":")[1].strip()
                        insight_score = float(lines[i + 2].split(":")[1].strip())
                        insight_category = lines[i + 3].split(":")[1].strip()
                        insight_description = lines[i + 4].split(":")[1].strip()
                        data_str = lines[i + 5]
                        index = data_str.index('Vega-Lite Json: ')
                        insight_vegalite = data_str[index + len('Vega-Lite Json: '):]
                        insight_list.append({"Header": header, "Type": insight_type, "Score": insight_score,
                                             "Category": insight_category, "Description": insight_description,
                                             "Vega-Lite": insight_vegalite})
                        i += 6
                    else:
                        i += 1
            else:
                i += 1
    return insight_list


def get_insight_vega_by_header(header_str, insight_list):
    header = eval(header_str)
    # sort for matching
    header = tuple(sorted(map(str, header)))

    insights_info = []
    for index, item in enumerate(insight_list):
        if item['Header'] == header:
            insight_info = {
                'realId': index,
                'type': item['Type'],
                'category': item['Category'],
                'score': item['Score'],
                'description': item['Description'],
                'vegaLite': item['Vega-Lite']
            }
            insights_info.append(insight_info)
    return insights_info


def get_top_k_insights(k, insight_list):
    top_k = sorted(enumerate(insight_list), key=lambda x: x[1]['Score'], reverse=True)[:k]
    insights_info = []
    for index, (real_id, item) in enumerate(top_k):
        insight_info = {
            'realId': real_id,
            'type': item['Type'],
            'category': item['Category'],
            'score': item['Score'],
            'description': item['Description'],
            'vegaLite': item['Vega-Lite']
        }
        insights_info.append(insight_info)
    return insights_info


def get_vega_lite_spec_by_id(id, insight_list):
    # id: insight id (node real-id)
    print(id)
    item = insight_list[id]
    vl_spec = item['Vega-Lite']
    print(vl_spec)
    return vl_spec


def get_insight_by_id(insight_list, id):
    # id: insight id (node real-id)
    item = insight_list[id]
    return item


table_structure = {
    'Company': ['Nintendo', 'Sony', 'Microsoft'],
    'Brand': ['Nintendo 3DS (3DS)', 'Nintendo DS (DS)', 'Nintendo Switch (NS)', 'Wii (Wii)', 'Wii U (WiiU)',
              'PlayStation 3 (PS3)', 'PlayStation 4 (PS4)', 'PlayStation Vita (PSV)', 'Xbox 360 (X360)',
              'Xbox One (XOne)'],
    'Location': ['Europe', 'Japan', 'North America', 'Other'],
    'Season': ['DEC', 'JUN', 'MAR', 'SEP'],
    'Year': ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
}


def convert_header_to_data_scope(header):
    data_scope = {
        'Company': '*',
        'Brand': '*',
        'Location': '*',
        'Season': '*',
        'Year': '*'
    }

    for value in header:
        for key, values_list in table_structure.items():
            if value in values_list:
                data_scope[key] = value
                break

    return data_scope


def convert_data_scope_to_header(data_scope):
    data_dict = json.loads(data_scope)
    header = []
    for key in ['Company', 'Brand', 'Location', 'Season', 'Year']:
        value = data_dict.get(key, '*')
        if value != '*':
            header.append(value)
    return tuple(header)


question2_prompt = """
Next, I will provide you with some other subspaces related to the current subspace in terms of header structure. 
"""


async def qa_LLM(query, item, insight_list, node_id):
    print("=====qa_LLM=====")
    question2 = combine_question2(query, item)
    print(f"combine_question2: \n{question2}\n\n")

    question3, categorized_headers = combine_question3(query, item)
    print(f"combine_question3: \n{question3}\n\n")
    # let LLM select one group that best matches the question and crt subspace
    response = get_response(question2 + question3)
    print(f"1. Response choose group: \n{response}\n\n")

    insights_info_dict, sort_insight_prompt = parse_response_select_group(response, query, insight_list)

    # let LLM sort insights
    response = get_response(sort_insight_prompt)
    print(f"2. Response sort insights: \n{response}\n\n")

    next_nodes, node_id = parse_response_select_insight(response, insights_info_dict, categorized_headers, node_id)

    print(f"next_nodes: {next_nodes}\n")
    print("=" * 100)

    return next_nodes, node_id

async def summarize_LLM(tree):
# def summarize_LLM(tree):
    print("=====summarize_LLM=====")
    prompt1 = """
You are given a data structure called an "insight tree," which consists of nodes and edges. Each node represents a data insight, and each edge describes the relationship between these insights. Your task is to generate a summary report that reflects **the user's exploration journey** through this insight tree.
Here is the insight tree:
Nodes:
- Each node has an `id`, a `type` that indicates the kind of insight (such as trend, outliers), a `description` that explains the data pattern of insight.
Edges:
- Each edge has:
  - a `source` and a `target`, representing the starting and ending nodes of a relationship. 
  - a `query`, which reflects the question or aspect that connects these insights.
  - a `relationship` that describes how the source insight leads to the target insight by semantically logical reasoning.
  - a `relType` indicating the structural type of relationship (same-level, generalization or specification).
Please analyze the provided insight tree and generate a coherent and logical report that outlines the user's exploration process, highlights significant insights discovered, and describes how the insights are interconnected. The report should also capture the logical reasoning behind the transitions from one insight to another.
Below is the JSON structure representing the insight tree:

    """

    tree_info = str(tree)

    prompt2 = """
   
Please note the following points:  
The purpose of this summary report is to help the user automatically summarize their exploration process and findings, effectively creating a data story. The report should seamlessly connect the insights (nodes) and relationships (edges) from the insight tree.
In the report, the 'query' represents the user's question or the aspect they want to explore further for a particular insight. The 'relationship' reflects the logical reasoning that led from a parent node to a child node as the next step in the exploration. The 'relType' indicates the structural relationship between two nodes, such as specification (a deeper exploration of a specific aspect), generalization (expanding to a broader context), or same-level (exploring another perspective within the same scope).
The summary report should carefully weave together all the nodes, including the user's queries and thoughts, into cohesive paragraphs. This insight tree may represent the user's analysis of a specific problem from several perspectives (as subtrees). You should emphasize reasoning about the user's thought process and understand what kind of data story this insight tree expresses.
When writing this report, use a data storytelling narrative style. Begin with an introduction to the user's initial area of interest, then develop the exploration by narrating how each insight was derived, and conclude by highlighting the overall insights discovered. For example, "The user was initially interested in X, and they sought to explore the reasons behind X by investigating Y..."
Ensure that the report is structured into well-formed paragraphs, avoiding lists or bullet points. Do not refer to the insights by their IDs in the report. Instead, use descriptive language to convey the insights and their relationships.
"""
    question = prompt1 + tree_info + prompt2

    # let LLM generate summary
    response = get_response(question)
    print(f"LLM Response Report Summary: \n{response}\n\n")

    # let LLM label the report
    prompt_label = """
You have generated a summary report that outlines a user's exploration journey through an insight tree. Now, the task is to enhance this report for interactive highlighting by embedding it with html <span> tags. These tags will associate specific sentences or phrases in the report with corresponding nodes in the insight tree. This allows the frontend to implement a feature where hovering over a report sentence highlights the associated elements in the insight tree.
To achieve this, follow these guidelines:
1. Identify Insight Nodes: For each sentence that describes or references a specific insight from the tree, wrap the **entire sentence** with a `<span>` tag and assign it a class using the format `<span class="insight-node-<id>">...</span>`, where `<id>` corresponds to the ID of the node in the insight tree.
2. The use of tags: Use as few `<span>` tags as possible. Use `<span>` tags to wrap as many sentences as possible, not words.
3. Selective Tagging: Not every sentence needs to be tagged. Only tag sentences that clearly correspond to a node in the insight tree.
4. Consistent Tagging: Ensure all insights and relationships mentioned in the report are tagged appropriately and consistently, providing comprehensive coverage for all nodes in the insight tree.
Here’s an example of how the report text should be annotated:
- Before: "The user observed a significant drop in Sony's sales, leading to further analysis."
- After: "<span class="insight-node-70">The user observed a significant drop in Sony's sales. This led to further analysis into the subsequent factors.</span>"

"""

    repeat_tree = """
I will repeat the insight tree below. 
Note: Please use the correct id (use the id in the insight tree).

    """

    report_input = """
Below is the report. Please annotate it according to the above guidelines and return the annotated report directly without any additional comments or text.

    """

    question2 = prompt_label + repeat_tree + tree_info + report_input + response
    final_report = get_response(question2)

    print(f"After labeling(The final report):\n {final_report}\n")
    print("=" * 100)

    return final_report


def combine_question2(query, item):
    crt_header = str(item['Header'])

    question2 = "Question: " + query + "\n"
    question2 += "Current Subspace: " + str(crt_header) + "\n"
    question2 += "Insight: \n"

    question2 += "Type: " + item['Type'] + "\n"
    question2 += "Score: " + str(item['Score']) + "\n"
    question2 += "Description: " + item['Description'] + "\n"
    question2 += question2_prompt

    return question2


def combine_question3(query, item):
    crt_header = str(item['Header'])

    # question contains only the related headers not insight-info
    question3 = """You already know the current data subspace and a problem that needs to be solved, and next we need to constantly \
change the data subspace to analyze the data. I will provide you with a "Related Subspaces List," \
which lists other subspaces related to the current subspace.
These subspaces are categorized into three types based on their hierarchical relationship with the current subspace: \
same-level, elaboration, and generalization. Please select a group that is most likely to solve my current problem \
as the next direction for exploration."""
    question3 += "Related Subspaces List:\n"
    grouping_string, categorized_headers = get_related_subspace(crt_header)
    question3 += grouping_string

    repeat_str = """Please note that my current subspace is: {} , and the question need to be solved is: "{}". \
Considering the subspace groups mentioned above, select one group that best matches the question."""
    repeat_str = repeat_str.format(str(crt_header), query)
    question3 += repeat_str

    response_format = """Your answer should follow the format below:
Group type: {}
Group Criteria: {}
Among them, Group type is used to identify the three categories of Same-level group, Elaboration group, and Generalization group, and Group Criteria is used to determine specific groups within the category.
For example:
Group type: Same-level groups
Group Criteria: Brand
Please answer strictly according to the format and do not add additional markings such as bold, punctuation marks, etc.
"""
    question3 += response_format
    return question3, categorized_headers


# test
# #
# insight_list = read_vis_list_into_insights('vis_list_VegaLite.txt')
#
# insight_id = 198
# query = "I want to know more about the sales in Japan of JUN."
# # query = "I want to know why the sale of the brand PlayStation 4 (PS4) is an outlier, what caused the unusually large value of this point?"
# #
# node_id = 0
# item = insight_list[insight_id]
# next_nodes = run(qa_LLM(query, item, insight_list, node_id))
#
# tree = """
# {'nodes': [
#     {'id': 265, 'type': 'outlier-temporal', 'description': 'The Sale of Year 2014 is an outlier, which is significantly higher than the normal Sale of other Years.', 'query': ''
#     },
#     {'id': 266, 'type': 'dominance', 'description': 'The Sale of PlayStation 4 (PS4) dominates among all Brands.', 'query': ''
#     },
#     {'id': 267, 'type': 'top2', 'description': 'The Sale proportion of Nintendo Switch (NS) and Nintendo 3DS (3DS) is significantly higher than that of other Brands.', 'query': ''
#     }
#   ], 'edges': [
#     {'source': 265, 'target': 266, 'query': None, 'relationship': '', 'relType': ''
#     },
#     {'source': 265, 'target': 267, 'query': None, 'relationship': '', 'relType': ''
#     }
#   ]
# }
# """
# summarize_LLM(tree)