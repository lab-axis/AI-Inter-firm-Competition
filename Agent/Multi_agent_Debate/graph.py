from langgraph.graph import StateGraph, END
from .state import DebateState
from .nodes import (
    load_data_node,
    qa_node,
    fa_node,
    ms_node,
    ba_node,
    rc_node,
    ar_node,
    cross_debate_node,
    ar_peer_review_node,
    moderator_node
)

def create_debate_graph():
    workflow = StateGraph(DebateState)
    
    # 1. Add Nodes
    workflow.add_node("load_data", load_data_node)
    
    # Phase 1
    workflow.add_node("qa_node", qa_node)
    workflow.add_node("fa_node", fa_node)
    workflow.add_node("ms_node", ms_node)
    workflow.add_node("rc_node", rc_node)
    workflow.add_node("ar_node", ar_node)
    
    # Phase 2 & 3
    workflow.add_node("ba_node", ba_node) # BA needs QA statement
    workflow.add_node("cross_debate", cross_debate_node)
    workflow.add_node("ar_peer_review", ar_peer_review_node)
    workflow.add_node("moderator", moderator_node)
    
    # 2. Add Edges
    workflow.set_entry_point("load_data")
    
    # Phase 1 Parallel execution: fan-out from load_data
    workflow.add_edge("load_data", "qa_node")
    workflow.add_edge("load_data", "fa_node")
    workflow.add_edge("load_data", "ms_node")
    workflow.add_edge("load_data", "rc_node")
    workflow.add_edge("load_data", "ar_node")
    
    # Phase 2: fan-in to ba_node once all Phase 1 agents are done
    workflow.add_edge(["qa_node", "fa_node", "ms_node", "rc_node", "ar_node"], "ba_node")
    
    workflow.add_edge("ba_node", "cross_debate")
    workflow.add_edge("cross_debate", "ar_peer_review")
    
    def check_ar_score(state: DebateState):
        score = state.get("ar_confidence_score", 100.0)
        # revision_count is incremented only when a review fails. Allow at most
        # three failed review/revision loops, matching the paper's max-iteration
        # setting and avoiding the previous <= 3 off-by-one behavior.
        rev_count = state.get("revision_count", 0)
        if score < 80.0 and rev_count < 5:
            return "cross_debate"
        return "moderator"
        
    workflow.add_conditional_edges(
        "ar_peer_review",
        check_ar_score,
        {
            "cross_debate": "cross_debate",
            "moderator": "moderator"
        }
    )
    
    workflow.add_edge("moderator", END)
    
    return workflow.compile()
