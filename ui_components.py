STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
#MainMenu, footer, header { visibility: hidden; }

.gitbot-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.5rem;
}
.gitbot-logo {
    width: 42px; height: 42px; background: #fc6d26;
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 22px; flex-shrink: 0;
}
.gitbot-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px; font-weight: 500; color: #e6edf3; margin: 0;
}
.gitbot-subtitle { font-size: 13px; color: #7d8590; margin: 2px 0 0; }

.chat-message {
    display: flex; gap: 12px;
    margin-bottom: 1.5rem; align-items: flex-start;
}
.chat-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace; font-weight: 500;
}
.avatar-user { background: #1f6feb; color: #e6edf3; }
.avatar-bot  { background: #fc6d26; color: #fff; }

.chat-bubble {
    flex: 1; padding: 12px 16px; border-radius: 0 12px 12px 12px;
    font-size: 14px; line-height: 1.7; color: #e6edf3;
    background: #161b22; border: 1px solid #21262d;
}
.bubble-user { border-radius: 12px 0 12px 12px; }
.bubble-bot ul  { margin: 8px 0 0 18px; padding: 0; }
.bubble-bot li  { margin-bottom: 4px; }
.bubble-bot strong { color: #fc6d26; }

.source-row {
    display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;
}
.source-badge {
    font-size: 11px; font-family: 'IBM Plex Mono', monospace;
    padding: 3px 10px; border-radius: 20px; border: 1px solid #30363d;
}
.badge-handbook  { background: #0d2137; color: #58a6ff; border-color: #1f6feb; }
.badge-direction { background: #1a1200; color: #e3b341; border-color: #9e6a03; }
.badge-chunks    { background: #161b22; color: #7d8590; border-color: #30363d; }

.error-box {
    background: #2d1b1b; border: 1px solid #f85149;
    border-radius: 10px; padding: 12px 16px;
    font-size: 13px; color: #f85149; margin-bottom: 1rem;
}

.stTextInput > div > div > input {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 10px !important; color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important; padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #fc6d26 !important;
    box-shadow: 0 0 0 3px rgba(252,109,38,0.15) !important;
}
.stButton > button {
    background: #fc6d26 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 14px !important;
    padding: 8px 20px !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
section[data-testid="stSidebar"] {
    background: #161b22 !important; border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
hr { border-color: #21262d !important; }
</style>
"""

HEADER_HTML = """
<div class="gitbot-header">
    <div class="gitbot-logo">🦊</div>
    <div>
        <div class="gitbot-title">GitBot</div>
        <div class="gitbot-subtitle">
            Trained on GitLab Handbook &amp; Product Direction
        </div>
    </div>
</div>
"""

EMPTY_STATE_HTML = """
<div style='text-align:center; padding:3rem 0; color:#7d8590;'>
    <div style='font-size:48px; margin-bottom:16px;'>🦊</div>
    <div style='font-family:"IBM Plex Mono",monospace; font-size:15px;
                color:#e6edf3; margin-bottom:8px;'>
        Ask me anything about GitLab
    </div>
    <div style='font-size:13px;'>
        Try a suggested question from the sidebar, or type your own below.
    </div>
</div>
"""


def user_message_html(content):
    return f"""
<div class="chat-message">
    <div class="chat-avatar avatar-user">U</div>
    <div class="chat-bubble bubble-user">{content}</div>
</div>"""


def bot_message_html(content, sources, chunk_count):
    badges = build_source_badges(sources, chunk_count)
    return f"""
<div class="chat-message">
    <div class="chat-avatar avatar-bot">G</div>
    <div class="chat-bubble bubble-bot">
        {content}
        {badges}
    </div>
</div>"""


def build_source_badges(sources, chunk_count):
    badges = ""
    if "handbook" in sources:
        badges += '<span class="source-badge badge-handbook">📘 GitLab Handbook</span>'
    if "direction" in sources:
        badges += '<span class="source-badge badge-direction">🗺️ Product Direction</span>'
    badges += f'<span class="source-badge badge-chunks">🔍 {chunk_count} chunks retrieved</span>'
    return f'<div class="source-row">{badges}</div>'


def error_box_html(message):
    return f'<div class="error-box">❌ {message}</div>'
