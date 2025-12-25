# GIRA System - Quick Start Guide

##  What Was Done

### 1. Cleaned Up Unused Files
Deleted the following unused/duplicate files:
- ❌ `gira-ai/gira-agent/document_upload/app/services/file_ingestor 2.py` (duplicate, 1259 lines)
- ❌ `gira-ai/gira-agent/document_upload/app/services/entity_extraction.py` (empty)
- ❌ `gira-ai/gira-agent/document_upload/app/services/manual_text_ingestor.py` (4 lines)
- ❌ `gira-ai/gira-agent/config/setting.py` (empty)
- ❌ `gira-ai/gira-mcp-server/config/prefetch_model.py` (empty)

### 2. Created Reference Documentation
-  **REFACTORING_SUMMARY.md** - Complete analysis of files needing refactoring
-  **ENV_TEMPLATE.md** - Template for all required environment variables
-  **check_file_sizes.ps1** - Script to monitor file sizes
-  **QUICK_START.md** - This file

### 3. Prepared Folder Structure
Created directories for future refactoring:
-  `gira-ai/gira-agent/pdf_highlighter/`
-  `gira-ai/gira-agent/database/services/`
-  `gira-ai/gira-agent/services/streaming/`
-  `gira-ai/gira-agent/services/mcp/`
-  `gira-ai/gira-agent/intent/`

## 🔧 Immediate Next Steps

### Step 1: Fix Docker Environment Variables ⚠️ CRITICAL

Your Docker services are failing due to missing environment variables. You need to:

1. **Generate FERNET_KEY:**
   ```powershell
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

2. **Update your `.env` file** with the variables from `ENV_TEMPLATE.md`

3. **Minimum required variables** to get started:
   ```env
   FERNET_KEY=<generated_key_from_step_1>
   GIRA_AGENT_INTERNAL_PORT=8081
   GIRA_AGENT_EXTERNAL_PORT=8081
   GIRA_BACKEND_INTERNAL_PORT=8082
   GIRA_BACKEND_EXTERNAL_PORT=8082
   GIRA_MCP_SERVER_INTERNAL_PORT=8085
   GIRA_MCP_SERVER_EXTERNAL_PORT=8085
   GIRA_FRONTEND_INTERNAL_PORT=3000
   GIRA_FRONTEND_EXTERNAL_PORT=3000
   GIRA_AGENT_DOCKER_URL=http://gira-agent:8081
   GIRA_FRONTEND_DOCKER_URL=http://gira-frontend:3000
   GIRA_API_BASE_URL=http://gira-backend:8082/api/v1
   ```

4. **Start Docker services:**
   ```powershell
   docker compose down
   docker compose up -d
   ```

5. **Check service status:**
   ```powershell
   docker compose ps
   docker compose logs --tail=50
   ```

### Step 2: Verify File Sizes

Run the file size check script:
```powershell
.\check_file_sizes.ps1
```

This will show you which files still need refactoring.

### Step 3: Priority Refactoring (Optional but Recommended)

**Highest Priority Files** (by size):
1. `pdf_highlighter.py` - 1,982 lines ⚠️
2. `file_ingestor.py` - 754 lines
3. `chunking_service.py` - 635 lines
4. `database/services.py` - 473 lines
5. `streaming_service.py` - 470 lines

See `REFACTORING_SUMMARY.md` for detailed refactoring plans for each file.

## 📊 Current System Stats

### Files Status:
- **Total Python files:** ~82 files (49 in gira-agent + 33 in gira-mcp-server)
- **Files > 300 lines:** 20 files need refactoring
- **Files 200-300 lines:** Several files (acceptable)
- **Files < 200 lines:** Majority of files (good)

### Largest Files:
1. `pdf_highlighter.py` - 1,982 lines
2. `file_ingestor 2.py` - 1,259 lines (DELETED )
3. `file_ingestor.py` - 754 lines
4. `chunking_service.py` - 635 lines
5. `database/services.py` - 473 lines

## 🎯 Goals

### Short Term (This Week)
- [x] Delete unused files
- [ ] Fix Docker environment variables
- [ ] Get all services running
- [ ] Refactor `pdf_highlighter.py` (most critical)

### Medium Term (This Month)
- [ ] Refactor all files > 500 lines
- [ ] Refactor all files 300-500 lines
- [ ] Achieve target: all files 200-300 lines maximum

### Long Term (Ongoing)
- [ ] Maintain file size discipline
- [ ] Regular code reviews
- [ ] Continuous refactoring

## 🛠️ Useful Commands

### Docker Management
```powershell
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View service status
docker compose ps

# View logs
docker compose logs --tail=50 <service-name>

# Restart a specific service
docker compose restart <service-name>

# Rebuild and restart
docker compose up -d --build <service-name>
```

### File Size Monitoring
```powershell
# Check file sizes
.\check_file_sizes.ps1

# Count lines in a specific file
(Get-Content <file-path> | Measure-Object -Line).Lines

# Find all files over 300 lines
Get-ChildItem -Path ".\gira-ai" -Recurse -Filter "*.py" | 
  ForEach-Object { 
    $lines = (Get-Content $_.FullName).Count; 
    if ($lines -gt 300) { "$lines`t$($_.FullName)" } 
  }
```

### Code Quality
```powershell
# Check Python syntax
python -m py_compile <file-path>

# Run all Python files through syntax check
Get-ChildItem -Path ".\gira-ai" -Recurse -Filter "*.py" | 
  ForEach-Object { 
    python -m py_compile $_.FullName 
  }
```

## 📚 Documentation Files

- **REFACTORING_SUMMARY.md** - Detailed analysis and refactoring plan
- **ENV_TEMPLATE.md** - Complete environment variable template
- **docker-compose.yml** - Docker service configuration
- **check_file_sizes.ps1** - File size analysis script

## ⚠️ Known Issues

1. **Docker services not starting** - Missing environment variables (see Step 1)
2. **Large files** - 20 files exceed 300-line target
3. **Port conflicts** - Ensure ports 3000, 8081, 8082, 8083, 8085, 9000, 9001, 5432 are available

## 🆘 Troubleshooting

### Docker services won't start
1. Check environment variables are set correctly
2. Check port conflicts: `netstat -ano | findstr ":<port>"`
3. Check logs: `docker compose logs <service-name>`
4. Try rebuilding: `docker compose up -d --build`

### Import errors after refactoring
1. Check `__init__.py` files exist in all packages
2. Update import paths in dependent files
3. Run syntax check: `python -m py_compile <file>`

### MinIO connection issues
1. Verify MinIO container is running: `docker compose ps minio`
2. Check MinIO credentials in env file
3. Ensure buckets are created: Check minio-init service logs

## 📞 Next Steps

1. **Immediate:** Fix Docker environment variables and get services running
2. **This Week:** Refactor `pdf_highlighter.py`
3. **This Month:** Refactor remaining large files
4. **Ongoing:** Maintain 200-300 line file size target

For detailed refactoring instructions for each file, see **REFACTORING_SUMMARY.md**.
