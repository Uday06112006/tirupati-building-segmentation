import uvicorn

if __name__ == "__main__":
    print("🚀 Starting OrthoGenerator 3D Reconstruction Microservice...")
    print("📖 Swagger API Docs: http://localhost:8002/docs")
    print("📊 Health Check:    http://localhost:8002/api/v1/health")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=False)
