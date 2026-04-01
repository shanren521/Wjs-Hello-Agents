"""赛博小镇 FastAPI 后端主程序"""

import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from models import (
    ChatRequest, ChatResponse,
    NPCStatusResponse, NPCListResponse, NPCInfo
)
from agents import get_npc_manager
from state_manager import get_state_manager

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    # 验证配置
    settings.validate()

    # 初始化NPC管理器
    npc_manager = get_npc_manager()

    # 初始化并启动状态管理器
    state_manager = get_state_manager()
    await state_manager.start()

    yield

    # 停止时
    await state_manager.stop()

# 创建FastAPI应用
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="赛博小镇 - 基于HelloAgents的API NPC对话系统",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取全局实例
npc_manager = None
state_manager = None

def get_managers():
    """获取管理实例"""
    global npc_manager, state_manager
    if npc_manager is None:
        npc_manager = get_npc_manager()
    if state_manager is None:
        state_manager = get_state_manager()

# ==============================API路由=========================
@app.get("/")
async def root():
    """根路径 - API信息"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "features": ["AI对话", "NPC记忆系统", "好感度系统", "批量状态更新"],
        "endpoints": {
            "docs": "/docs",
            "chat": "/chat",
            "npcs": "/npcs",
            "npcs_status": "/npcs/status",
            "npc_memories": "/npcs/{npc_name}/memories",
            "npc_affinity": "/npcs/{npc_name}/affinity",
            "all_affinities": "/affinities"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": "now"}

@app.post("/chat", response_model=ChatResponse)
async  def chat_with_npc(request: ChatRequest):
    """
    与NPC对话接口
    玩家与指定npc进行实时对话，使用独立的Agent处理
    :param request:
    :return:
    """

    npc_mgr, _ = get_managers()

    # 验证NPC是否存在
    npc_info = npc_mgr.get_npc_info(request.npc_name)
    if not npc_info:
        raise HTTPException(status_code=404, detail=f"NPC {request.npc_name}不存在")

    try:
        # 调用NPC Agent处理对话
        response_text = npc_mgr.chat(request.npc_name, request.message)
        return ChatResponse(
            npc_name=request.npc_name,
            message=response_text,
            npc_title=npc_info["title"],
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"对话处理失败：{str(e)}"
        )

@app.get("/npcs", response_model=NPCListResponse)
async def list_npcs():
    """获取所有NPC列表"""
    npc_mgr, _ = get_managers()

    npcs_data = npc_mgr.get_all_npcs()
    npcs = [NPCInfo(**npc) for npc in npcs_data]

    return NPCListResponse(
        npcs=npcs,
        total=len(npcs)
    )

@app.get("/npcs/status", response_model=NPCStatusResponse)
async def get_npcs_status():
    """获取所有NPC的当前状态
    返回批量生成的NPC对话内容，用于显示NPC的自主行为
    """
    _, state_mgr = get_managers()

    state = state_mgr.get_current_state()

    return NPCStatusResponse(
        dialogues=state["dialogue"],
        last_update=state["last_update"],
        next_update_in=state["next_update_in"]
    )

@app.post("/npcs/status/refresh")
async def refresh_npcs_status():
    """
    强制刷新NPC状态
    立即触发一次批量对话生成
    :return:
    """
    _, state_mgr = get_managers()

    await state_mgr.force_update()
    state = state_mgr.get_current_state()

    return {
        "message": "NPC状态已刷新",
        "dialogues": state["dialogue"]
    }

@app.get("/npcs/{npc_name}")
async def get_npc_info(npc_name: str):
    """获取指定NPC的详细信息"""
    npc_mgr, state_mgr = get_managers()

    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(status_code=404, detail=f"NPC {npc_name}不存在")

    # 添加当前对话
    current_dialogue = state_mgr.get_npc_dialogue(npc_name)
    npc_info["current_dialogue"] = current_dialogue

    return npc_info

@app.get("/npcs/{npc_name}/memories")
async def get_npc_memories(npc_name: str, limit: int = 10):
    """获取NPC的记忆列表
    Args:
        npc_name: NPC名称
        limit: 返回的记忆数量限制
    Returns:
        NPC的记忆列表
    """
    npc_mgr, _ = get_managers()

    # 验证NPC是否存在
    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(status_code=404, detail=f"NPC {npc_name}不存在")

    try:
        memories = npc_mgr.get_npc_memories(npc_name, limit=limit)
        return {
            "npc_name": npc_name,
            "memories": memories,
            "total": len(memories)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取记忆列表失败：{str(e)}"
        )



# ============================主程序入口=========================
if __name__ == "__main__":
    print("\n 启动赛博小镇后端服务...")
    print(f"监听地址: {settings.API_HOST}: {settings.API_PORT}")

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )










