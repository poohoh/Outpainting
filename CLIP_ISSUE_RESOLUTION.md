# CLIP Loading Issue Resolution Report

**Date**: 2025-08-09  
**Issue**: OSError when loading CLIP tokenizer in outpainting model  
**Status**: ✅ RESOLVED

## 🚨 Original Problem

### Error Message
```
OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'. 
If you were trying to load it from 'https://huggingface.co/models', 
make sure you don't have a local directory with the same name. 
Otherwise, make sure 'openai/clip-vit-large-patch14' is the correct path 
to a directory containing all relevant files for a CLIPTokenizer tokenizer.
```

### Error Location
- **File**: `/app/ldm/modules/encoders/modules.py`
- **Class**: `FrozenCLIPEmbedder.__init__()`
- **Line**: 103 - `self.tokenizer = CLIPTokenizer.from_pretrained(version)`

### Root Cause Analysis
1. **Initialization Order Issue**: 
   - Model architecture creation → CLIP download attempt → Checkpoint loading
   - HuggingFace download was required during model instantiation, before checkpoint loading
   
2. **Network/Cache Issue**:
   - No local HuggingFace cache available
   - `from_pretrained()` attempting remote download but failing

## 🔍 Investigation Process

### 1. Environment Analysis
- **Python Environment**: `/opt/conda/envs/pytorch-legacy`
- **Transformers Version**: 4.19.2
- **Network Connectivity**: ✅ Working (verified with curl)
- **HuggingFace Access**: ✅ Available

### 2. Checkpoint Analysis
```python
# Found CLIP weights already present in checkpoint
ckpt['state_dict'] contains:
- 'cond_stage_model.transformer.text_model.*' (197 parameters)
- Complete CLIP text encoder weights
```

### 3. Web Research Results
- Common issue across multiple Stable Diffusion projects
- Main solutions: cache clearing, manual download, offline mode
- Confirmed that direct path loading works better than HuggingFace hub

## ⚡ Solution Implementation

### Step 1: Manual Model Download
Downloaded all required CLIP model files to `/app/models/openai--clip-vit-large-patch14/`:

```bash
# Downloaded files:
- config.json (4,519 bytes)
- tokenizer.json (2,224,003 bytes) 
- tokenizer_config.json (905 bytes)
- vocab.json (961,143 bytes)
- merges.txt (524,619 bytes)
- pytorch_model.bin (1,710,671,599 bytes)
```

### Step 2: Code Modification
**File**: `/app/ldm/modules/encoders/modules.py`

**Before**:
```python
def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
             freeze=True, layer="last", layer_idx=None):
    super().__init__()
    assert layer in self.LAYERS
    self.tokenizer = CLIPTokenizer.from_pretrained(version)
    self.transformer = CLIPTextModel.from_pretrained(version)
```

**After**:
```python
def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
             freeze=True, layer="last", layer_idx=None):
    super().__init__()
    assert layer in self.LAYERS
    
    # Try local path first, then fallback to HuggingFace
    local_path = "/app/models/openai--clip-vit-large-patch14"
    try:
        self.tokenizer = CLIPTokenizer.from_pretrained(local_path)
        self.transformer = CLIPTextModel.from_pretrained(local_path)
        print(f"✅ Loaded CLIP from local path: {local_path}")
    except:
        print(f"⚠️ Local path failed, trying HuggingFace: {version}")
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
```

### Step 3: Verification Testing
```python
# Test result:
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained('/app/models/openai--clip-vit-large-patch14')
# ✅ SUCCESS: Tokenizer loaded successfully!
# Vocab size: 49408
# Test tokenization: 7 tokens for "a photo of a dog"
```

## ✅ Final Results

### Success Indicators
1. **CLIP Loading**: ✅ `Loaded CLIP from local path: /app/models/openai--clip-vit-large-patch14`
2. **Model Instantiation**: ✅ FrozenCLIPEmbedder created successfully
3. **Tokenization**: ✅ Working (vocab_size: 49408, max_length: 77)

### Test Script Output
```bash
cd /app && python test/check_unet_load.py

✅ Loaded CLIP from local path: /app/models/openai--clip-vit-large-patch14
No module 'xformers'. Proceeding without it.
LatentInpaintDiffusion: Running in eps-prediction mode
DiffusionWrapper has 859.54 M params.
# ... model continues loading (CLIP issue resolved)
```

## 📋 Current Status

### ✅ Resolved Issues
- [x] CLIP tokenizer loading error
- [x] FrozenCLIPEmbedder initialization
- [x] HuggingFace download dependency removed
- [x] Local model path integration

### ⚠️ Remaining Issues (Separate from CLIP)
- EMA (Exponential Moving Average) weight mismatches in checkpoint loading
- These are model architecture/checkpoint compatibility issues, not CLIP-related

## 🔄 Improvement Recommendations

### 1. Robustness Enhancement
Consider adding more comprehensive error handling:
```python
def safe_clip_loading(local_path, fallback_version):
    """Enhanced CLIP loading with multiple fallback strategies"""
    strategies = [
        lambda: CLIPTokenizer.from_pretrained(local_path),
        lambda: CLIPTokenizer.from_pretrained(fallback_version, cache_dir='/app/cache'),
        lambda: CLIPTokenizer.from_pretrained(fallback_version, local_files_only=True),
    ]
    
    for strategy in strategies:
        try:
            return strategy()
        except Exception as e:
            logging.warning(f"CLIP loading strategy failed: {e}")
    
    raise RuntimeError("All CLIP loading strategies failed")
```

### 2. Configuration Management
Add CLIP model path to configuration files for easier management.

### 3. Documentation
Update project documentation to include CLIP model setup instructions.

## 📁 Files Modified

1. **`/app/ldm/modules/encoders/modules.py`**
   - Modified `FrozenCLIPEmbedder.__init__()` method
   - Added local path fallback logic
   - Added informative logging

2. **New Directory Structure**:
   ```
   /app/models/openai--clip-vit-large-patch14/
   ├── config.json
   ├── merges.txt
   ├── pytorch_model.bin
   ├── tokenizer.json
   ├── tokenizer_config.json
   └── vocab.json
   ```

## 🎯 Key Learnings

1. **Local Model Storage**: Storing models locally eliminates network dependencies
2. **Fallback Strategies**: Graceful degradation improves reliability
3. **Error Context**: Understanding the initialization order was crucial
4. **Web Research**: Community solutions provided valuable insights
5. **Verification**: Systematic testing confirmed the fix worked

---

**Resolution completed successfully on 2025-08-09**
**Next steps**: Address remaining EMA weight issues for complete model loading