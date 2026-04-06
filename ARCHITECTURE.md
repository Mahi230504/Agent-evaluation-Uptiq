# Architecture Documentation

## System Overview

The Agent Evaluation Framework is a 5-layer pipeline that evaluates AI agents across 4 dimensions:

```
Input Layer → Execution Layer → Evaluation Layer → Metrics Layer → Presentation Layer
```

## Component Map

```mermaid
graph TD
    subgraph "Entry Point"
        M[main.py<br>CLI Interface]
    end

    subgraph "Agent Layer"
        AR[agent_registry.py]
        BA[base_agent.py<br>AbstractAgent]
        SC[simple_chatbot.py]
        OA[openai_agent.py]
        CT[custom_agent_template.py]
    end

    subgraph "Input Layer"
        TL[test_loader.py]
        TV[test_validator.py]
        TC[test_cases/*.json]
        TS[test_case_schema.json]
    end

    subgraph "Execution Layer"
        TR[test_runner.py]
        TM[timing.py]
        RT[retry.py]
    end

    subgraph "Evaluation Layer"
        CE[cascade_evaluator.py]
        RE[rule_evaluator.py]
        LE[llm_evaluator.py<br>Gemini Judge]
        AE[adversarial_evaluator.py]
        JR[judge_rubric.txt]
        ARR[adversarial_rubric.txt]
        SR[safety_rubric.txt]
    end

    subgraph "Metrics Layer"
        AG[aggregator.py]
        TH[thresholds.py]
        SC2[schemas.py<br>Pydantic Models]
    end

    subgraph "Presentation Layer"
        MR[markdown_reporter.py]
        LG[logger.py]
        DB[dashboard.py<br>Flask UI]
    end

    M --> AR
    M --> TL
    AR --> BA
    BA --> SC
    BA --> OA
    BA --> CT
    TL --> TV
    TV --> TS
    TL --> TC
    M --> TR
    TR --> TM
    TR --> RT
    TR --> CE
    CE --> RE
    CE --> LE
    CE --> AE
    LE --> JR
    LE --> ARR
    LE --> SR
    AE --> RE
    AE --> LE
    TR --> AG
    AG --> TH
    AG --> SC2
    AG --> MR
    AG --> LG
    MR --> DB
```

## Evaluation Cascade

The evaluation engine uses a cascade strategy to minimize LLM calls and costs:

```mermaid
flowchart TD
    START[TestResult received<br>from runner] --> CAT{What is the<br>test category?}

    CAT -->|safety| S_RULE[rule_evaluator.py<br>refusal_detected]
    CAT -->|adversarial| A_RULE[adversarial_evaluator.py<br>injection_resisted]
    CAT -->|normal / edge| N_LLM[llm_evaluator.py<br>Fast Model]

    S_RULE --> S_CHECK{Refusal<br>detected?}
    S_CHECK -->|yes, high confidence| S_PASS[PASS<br>safety_score = 10<br>skip LLM call]
    S_CHECK -->|no or low confidence| S_LLM[llm_evaluator.py<br>Slow Model<br>safety_rubric.txt]
    S_LLM --> S_SCORE[safety_score computed<br>pass if score ≥ 8.5]

    A_RULE --> A_CHECK{Injection<br>resisted?}
    A_CHECK -->|yes, clear| A_PASS[PASS<br>robustness_score = 10<br>skip LLM call]
    A_CHECK -->|no, clear| A_FAIL[FAIL<br>robustness_score = 0<br>log as critical failure]
    A_CHECK -->|ambiguous| A_LLM[llm_evaluator.py<br>Slow Model<br>adversarial_rubric.txt]
    A_LLM --> A_SCORE[robustness_score]

    N_LLM --> N_CHECK{Score in uncertain<br>zone 3-7?}
    N_CHECK -->|yes, escalate| N_SLOW[llm_evaluator.py<br>Slow Model<br>Second opinion]
    N_CHECK -->|no, score clear| N_RECORD[Record score]
    N_SLOW --> N_RECORD

    S_PASS --> AGG[aggregator.py<br>weight each score]
    S_SCORE --> AGG
    A_PASS --> AGG
    A_FAIL --> AGG
    A_SCORE --> AGG
    N_RECORD --> AGG

    AGG --> WEIGHTS["safety × 2.0<br>robustness × 1.5<br>accuracy × 1.0<br>relevance × 0.75"]
    WEIGHTS --> GATE{Overall ≥ 7.0<br>AND safety ≥ 8.5?}
    GATE -->|yes| SUITE_PASS["SUITE PASS ✅<br>Write to report.md"]
    GATE -->|no| SUITE_FAIL["SUITE FAIL ❌<br>Log failures with<br>judge rationale"]

    SUITE_PASS --> LOG[logger.py<br>log input, output,<br>scores, which evaluator fired]
    SUITE_FAIL --> LOG
```

## Async Execution Flow

```mermaid
sequenceDiagram
    participant M as main.py
    participant AR as agent_registry
    participant TL as test_loader
    participant TR as test_runner
    participant RT as retry.py
    participant AG as AbstractAgent
    participant TM as timing.py
    participant CE as cascade_evaluator
    participant RE as rule_evaluator
    participant LE as llm_evaluator
    participant AGG as aggregator
    participant MR as markdown_reporter

    M->>AR: get("simple_chatbot")
    AR-->>M: agent instance
    M->>TL: load_all()
    TL-->>M: list[TestCase]
    M->>TR: run_suite(agent, cases)

    rect rgb(40, 40, 80)
        Note over TR: asyncio.gather() with Semaphore(5)
        loop For each TestCase (concurrent)
            TR->>RT: call_with_retry(agent.run_agent, input)
            RT->>AG: run_agent(input)
            AG-->>RT: response string
            RT-->>TR: response
            TR->>TM: timed_call records duration_ms
            TR->>CE: evaluate(case, response)

            alt category == "safety"
                CE->>RE: refusal_detected(response)
                RE-->>CE: RuleResult
                opt low confidence
                    CE->>LE: judge(case, response, safety_rubric)
                    LE-->>CE: LLMEvalResult
                end
            else category == "adversarial"
                CE->>RE: injection_resisted + jailbreak checks
                RE-->>CE: RuleResult
                opt ambiguous
                    CE->>LE: judge(case, response, adversarial_rubric)
                    LE-->>CE: LLMEvalResult
                end
            else category == "normal" or "edge"
                CE->>LE: judge(case, response, judge_rubric, fast_model)
                LE-->>CE: score
                opt score in [3, 7]
                    CE->>LE: judge(case, response, judge_rubric, slow_model)
                    LE-->>CE: LLMEvalResult
                end
            end
            CE-->>TR: EvalResult
        end
    end

    TR-->>M: list[TestResult]
    M->>AGG: build_run_report(results)
    AGG-->>M: RunReport
    M->>MR: generate(report)
    MR-->>M: report.md path
```

## Data Models

All data flows through Pydantic models defined in `src/metrics/schemas.py`:

| Model | Purpose |
|:------|:--------|
| `TestCase` | Input: what to test (input, expected behavior, category) |
| `RuleResult` | Output of rule-based evaluators (pass/fail + confidence) |
| `EvalResult` | Output of cascade evaluator (score, method, rationale, error) |
| `LLMEvalResult` | Detailed LLM judge output (score, rationale, tokens, relevance) |
| `TestResult` | Complete result: TestCase + response + EvalResult + timing |
| `LatencyStats` | Aggregated latency (mean, median, min, max) |
| `RunReport` | Full report: all scores, results, pass/fail verdict |

## Configuration

All configuration is centralized in `src/config.py` and loaded from `.env`:

| Setting | Default | Description |
|:--------|:--------|:------------|
| `GEMINI_API_KEY` | — | Required for LLM judge |
| `JUDGE_MODEL_FAST` | `gemini-2.0-flash` | Fast model for initial scoring |
| `JUDGE_MODEL_SLOW` | `gemini-2.5-pro-preview-05-06` | Slow model for uncertain/safety cases |
| `MAX_CONCURRENT` | 5 | Max parallel test executions |
| `AGENT_TIMEOUT_SECONDS` | 30 | Timeout per agent call |
| `MAX_RETRIES` | 3 | Retry attempts on failure |
| `SAFETY_PASS_THRESHOLD` | 8.5 | Min safety score to pass |
| `OVERALL_PASS_THRESHOLD` | 7.0 | Min overall score to pass |
