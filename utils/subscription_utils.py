from datetime import date, datetime
from models import Subscription, Tool, ToolUsage, User, Plan , db
from flask import session
import json

def has_active_subscription(user_id):
    subscription = Subscription.query.filter_by(user_id=user_id, status='Active').filter(Subscription.start_date <= date.today(), Subscription.end_date >= date.today()).first()
    return subscription is not None

def is_trial_available(user_id, tool_name):
    tool = Tool.query.filter_by(name=tool_name).first()
    if not tool:
        return False, {"status": "error", "message": "Tool not found"}

    tool_id = tool.id

    tool_usage = ToolUsage.query.filter_by(user_id=user_id, tool_id=tool_id).first()
    if not tool_usage:
        return True, 0  # No trials used yet

    remaining_trials = tool_usage.max_trials
    return remaining_trials > 0, remaining_trials

def increment_tool_usage(user_id, tool_name, duration):
    tool = Tool.query.filter_by(name=tool_name).first()
    if not tool:
        return {"status": "error", "message": "Tool not found"}

    tool_id = tool.id

    tool_usage = ToolUsage.query.filter_by(user_id=user_id, tool_id=tool_id).first()
    if tool_usage and tool_usage.max_trials > 0:
        tool_usage.usage_count += duration
        tool_usage.max_trials -= duration
        tool_usage.last_used = datetime.utcnow()  # Update last_used with the current timestamp
        db.session.commit()
        return {"status": "success", "message": "Usage updated successfully"}
    else:
        return {"status": "error", "message": "No remaining trials or record not found"}, 404

def get_trials_left():
    user_id = session['user_id']
    tool_name = "video-bg-remover"

    tool = Tool.query.filter_by(name=tool_name).first()
    if not tool:
        return {"status": "error", "message": "Tool not found"}, 404

    tool_id = tool.id

    tool_usage = ToolUsage.query.filter_by(user_id=user_id, tool_id=tool_id).first()
    if not tool_usage:
        user = User.query.get(user_id)
        if not user:
            return {"status": "error", "message": "User not found"}, 404

        plan = Plan.query.get(user.plan_id)
        if not plan:
            return {"status": "error", "message": "Plan not found"}, 404

        trials_per_tool = json.loads(plan.trials_per_tool)
        max_trials = trials_per_tool.get(tool_name, 0)

        tool_usage = ToolUsage(user_id=user_id, tool_id=tool_id, usage_count=0, max_trials=max_trials)
        db.session.add(tool_usage)
        db.session.commit()
        return max_trials

    return tool_usage.max_trials