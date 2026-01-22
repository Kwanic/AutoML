# Literature Reviews

AI-generated research summaries using GPT-5 web search for model selection.

## Structure

```
literature_reviews/
└── literature_review_{data_type}_{timestamp}.json
```

Each JSON contains:
- `query`: Research query sent to GPT-5
- `review_text`: Comprehensive summary
- `key_findings`: Important insights from recent papers
- `recommended_approaches`: Suggested model architectures
- `recent_papers`: Citations and contributions
- `confidence`: Review confidence score (0-1)

Reviews are automatically generated when `SKIP_LITERATURE_REVIEW=false` in `.env`.
Disable to save API costs during development.
