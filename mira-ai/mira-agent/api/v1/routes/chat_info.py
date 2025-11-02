


# @app.get("/get_user_chat_sessions")
# async def get_user_chat_sessions(
#     request: Request,
#     limit: int = 50
# ):
#     # Decode user ID from headers
#     final_user_id, country = decode_user_id_from_header(request)

#     if not final_user_id:
#         return JSONResponse(
#             content={"error": "User ID not found in headers"}, 
#             status_code=400
#         )
    
#     try:
#         data = await DatabaseService.get_user_chat_sessions(
#             user_id=final_user_id,
#             limit=limit,
#         )
#         return JSONResponse(content=data, status_code=200)
    
#     except Exception as e:
#         print(f"[get_user_chat_sessions] Error retrieving chat sessions: {e}")
#         return JSONResponse(content={"error": "Failed to retrieve chat sessions"}, status_code=500)
