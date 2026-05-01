package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
)

type Request struct {
	Task string `json:"task"`
	Text string `json:"text"`
}

type Signal struct {
	Label  string  `json:"label"`
	Score  float64 `json:"score"`
	Reason string  `json:"reason"`
}

type Response struct {
	Engine     string   `json:"engine"`
	Runtime    string   `json:"runtime"`
	Label      string   `json:"label"`
	Confidence float64  `json:"confidence"`
	Tokens     []string `json:"tokens"`
	Signals    []Signal `json:"signals"`
	Reply      string   `json:"reply"`
}

var tokenPattern = regexp.MustCompile(`[a-zA-Z0-9_]+`)

var intents = map[string]map[string]float64{
	"code": {
		"api": 1.3, "backend": 1.4, "frontend": 1.2, "bug": 1.4, "error": 1.2,
		"fix": 1.1, "function": 1.1, "model": 0.8, "python": 1.5, "rust": 1.5,
		"go": 1.4, "cpp": 1.5, "typescript": 1.3, "database": 1.1,
	},
	"learning": {
		"learn": 1.6, "belajar": 1.6, "explain": 1.4, "explanation": 1.4,
		"why": 1.1, "how": 1.1, "tutorial": 1.5, "konsep": 1.3, "materi": 1.2,
	},
	"idea": {
		"idea": 1.4, "design": 1.2, "plan": 1.2, "brainstorm": 1.5, "buat": 0.8,
		"build": 0.9, "fitur": 1.2, "produk": 1.1,
	},
	"data": {
		"data": 1.5, "dataset": 1.6, "csv": 1.4, "feature": 1.1, "train": 1.4,
		"accuracy": 1.3, "predict": 1.2, "classification": 1.2,
	},
}

var sentiment = map[string]map[string]float64{
	"positive": {
		"good": 1.0, "great": 1.3, "clean": 1.1, "simple": 1.0, "bagus": 1.2,
		"suka": 1.2, "mantap": 1.3, "cepat": 0.8,
	},
	"negative": {
		"bad": 1.0, "error": 0.8, "broken": 1.2, "susah": 1.0, "bingung": 1.1,
		"gagal": 1.2, "lambat": 1.0, "jelek": 1.2,
	},
}

func main() {
	body, err := io.ReadAll(os.Stdin)
	if err != nil {
		fail(err)
	}

	var req Request
	if err := json.Unmarshal(body, &req); err != nil {
		fail(err)
	}

	resp := analyze(req.Text)
	out, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		fail(err)
	}

	fmt.Println(string(out))
}

func analyze(text string) Response {
	tokens := tokenize(text)
	intentSignals := scoreGroups(tokens, intents)
	sentimentSignals := scoreGroups(tokens, sentiment)

	label := "general"
	confidence := 0.45
	if len(intentSignals) > 0 && intentSignals[0].Score > 0 {
		label = intentSignals[0].Label
		confidence = clamp(0.48+math.Tanh(intentSignals[0].Score/4)*0.42, 0.05, 0.96)
	}

	signals := append([]Signal{}, intentSignals...)
	signals = append(signals, sentimentSignals...)
	sort.Slice(signals, func(i, j int) bool {
		return signals[i].Score > signals[j].Score
	})

	return Response{
		Engine:     "go_text_ai",
		Runtime:    "go",
		Label:      label,
		Confidence: round(confidence, 3),
		Tokens:     topTokens(tokens, 14),
		Signals:    trimSignals(signals, 6),
		Reply:      buildReply(label, sentimentSignals),
	}
}

func tokenize(text string) []string {
	raw := tokenPattern.FindAllString(strings.ToLower(text), -1)
	tokens := make([]string, 0, len(raw))
	for _, token := range raw {
		if len(token) < 2 {
			continue
		}
		tokens = append(tokens, token)
	}
	return tokens
}

func scoreGroups(tokens []string, groups map[string]map[string]float64) []Signal {
	counts := map[string]int{}
	for _, token := range tokens {
		counts[token]++
	}

	signals := make([]Signal, 0, len(groups))
	for label, weights := range groups {
		score := 0.0
		reasons := []string{}
		for token, count := range counts {
			if weight, ok := weights[token]; ok {
				score += weight * float64(count)
				reasons = append(reasons, token)
			}
		}
		sort.Strings(reasons)
		signals = append(signals, Signal{
			Label:  label,
			Score:  round(score, 3),
			Reason: strings.Join(reasons, ", "),
		})
	}

	sort.Slice(signals, func(i, j int) bool {
		return signals[i].Score > signals[j].Score
	})
	return signals
}

func topTokens(tokens []string, limit int) []string {
	counts := map[string]int{}
	for _, token := range tokens {
		counts[token]++
	}

	unique := make([]string, 0, len(counts))
	for token := range counts {
		unique = append(unique, token)
	}
	sort.Slice(unique, func(i, j int) bool {
		if counts[unique[i]] == counts[unique[j]] {
			return unique[i] < unique[j]
		}
		return counts[unique[i]] > counts[unique[j]]
	})

	if len(unique) > limit {
		return unique[:limit]
	}
	return unique
}

func trimSignals(signals []Signal, limit int) []Signal {
	filtered := []Signal{}
	for _, signal := range signals {
		if signal.Score <= 0 {
			continue
		}
		filtered = append(filtered, signal)
	}
	if len(filtered) > limit {
		return filtered[:limit]
	}
	return filtered
}

func buildReply(label string, sentimentSignals []Signal) string {
	tone := "neutral"
	if len(sentimentSignals) > 0 && sentimentSignals[0].Score > 0 {
		tone = sentimentSignals[0].Label
	}

	switch label {
	case "code":
		if tone == "negative" {
			return "Break the issue into input, process, and output, then test the smallest failing path."
		}
		return "Start with a small endpoint or function, measure the output, then iterate."
	case "learning":
		return "Turn the topic into one concept, one example, and one small experiment."
	case "idea":
		return "Pick the smallest version that proves the core behavior."
	case "data":
		return "Check the dataset shape, target label, baseline score, then model changes."
	default:
		return "Ask a narrower question or add sample input for a stronger result."
	}
}

func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func round(value float64, places int) float64 {
	scale := math.Pow(10, float64(places))
	return math.Round(value*scale) / scale
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
