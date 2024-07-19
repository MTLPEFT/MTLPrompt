## Config

```
  MTLPROMPT:
    ENABLED: True
    DECODER_TYPE: "baseline" # " baseline : MTLoRA Decoder" 
    PROMPT:
      PROMPT_DROPOUT: 0
      SHARED:
        TYPE: SHALLOW # " DEEP | SHALLOW "
        LEN: 10   # 
        METHOD: "prepend" 
      SPATIAL:
        ENABLED: True # "Use task-specific prompt in encoder stage's last block (Similar to baseline)"
        METHOD: "prepend" # prepend | low-rank
        TYPE: SHALLOW # " DEEP | SHALLOW "
        LEN: 10  #  TODO : 
```
