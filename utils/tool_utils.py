from models import Tool

def get_tool_id(tool_name):
    tool = Tool.query.filter_by(name=tool_name).first()
    return tool.id if tool else None