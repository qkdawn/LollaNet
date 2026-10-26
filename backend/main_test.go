package main

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSnapshotAPI(t *testing.T) {
	repo, err := LoadRepository("data/library.json")
	if err != nil {
		t.Fatal(err)
	}
	router := Router(repo)
	for _, path := range []string{"/api/meta", "/api/categories", "/api/models", "/api/modes", "/api/associations"} {
		w := httptest.NewRecorder()
		router.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != 200 || !json.Valid(w.Body.Bytes()) {
			t.Fatalf("%s: %d %s", path, w.Code, w.Body.String())
		}
	}
	model := repo.library.Models[0]
	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest("GET", "/api/models/"+model.ID, nil))
	var found Model
	if err := json.Unmarshal(w.Body.Bytes(), &found); err != nil || found.ID != model.ID {
		t.Fatal("detail mismatch")
	}
	for _, path := range []string{"/api/models/no-such-model", "/api/modes/no-such-mode", "/missing"} {
		w := httptest.NewRecorder()
		router.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != 404 {
			t.Fatalf("expected 404: %s", path)
		}
	}
	for _, filter := range []Filters{{Query: strings.ToUpper(model.EnglishName)}, {Category: strings.Join(model.CategoryPath[:2], "/")}, {Status: model.Status, Kind: model.Kind}} {
		results := repo.Models(filter)
		if len(results) == 0 {
			t.Fatalf("empty filter: %+v", filter)
		}
		for _, result := range results {
			if filter.Status != "" && result.Status != filter.Status || filter.Kind != "" && result.Kind != filter.Kind {
				t.Fatal("filter mismatch")
			}
		}
	}
	if len(repo.Models(Filters{Query: "__does_not_exist__"})) != 0 {
		t.Fatal("expected empty query results")
	}
	if len(repo.Models(Filters{Category: "学科"})) != 0 {
		t.Fatal("category must match path segment")
	}
}
func TestInvalidSnapshots(t *testing.T) {
	if _, err := LoadRepository(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("missing file accepted")
	}
	for _, contents := range []string{"broken", `{}`, `{"schemaVersion":2}`, `{"schemaVersion":1,"generatedAt":"today"}`} {
		filename := filepath.Join(t.TempDir(), "invalid.json")
		if err := os.WriteFile(filename, []byte(contents), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadRepository(filename); err == nil {
			t.Fatal("invalid snapshot accepted")
		}
	}
}

func TestReadingsAndSummarySearch(t *testing.T) {
	repo, err := LoadRepository("data/library.json")
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, model := range repo.library.Models {
		if model.Reading != nil {
			count++
			if model.Reading.ModelID != model.ID || model.Reading.ModelVersion != model.Version {
				t.Fatal("reading version mismatch")
			}
		}
	}
	batchData, err := os.ReadFile("../frontend/content/reading-batches.json")
	if err != nil {
		t.Fatal(err)
	}
	var batches []struct {
		ModelIDs []string `json:"modelIds"`
	}
	if err := json.Unmarshal(batchData, &batches); err != nil {
		t.Fatal(err)
	}
	expected := map[string]bool{}
	for _, batch := range batches {
		for _, id := range batch.ModelIDs {
			if expected[id] {
				t.Fatalf("duplicate reading batch entry: %s", id)
			}
			expected[id] = true
		}
	}
	if count != len(expected) {
		t.Fatalf("expected %d readings, got %d", len(expected), count)
	}
	for _, model := range repo.library.Models {
		if (model.Reading != nil) != expected[model.ID] {
			t.Fatalf("reading batch mismatch: %s", model.ID)
		}
	}
	var results []Model
	// This wording exists in the new summary, not the legacy definition or title.
	w := httptest.NewRecorder()
	Router(repo).ServeHTTP(w, httptest.NewRequest("GET", "/api/models?q=%E4%B8%80%E5%8D%83%E4%BA%BA%E7%9A%84%E8%B4%A6", nil))
	if err := json.Unmarshal(w.Body.Bytes(), &results); err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != "mm-base-rate" || results[0].Reading == nil {
		t.Fatal("reading summary search failed")
	}
	w = httptest.NewRecorder()
	Router(repo).ServeHTTP(w, httptest.NewRequest("GET", "/api/models/mm-base-rate", nil))
	var detail Model
	if err := json.Unmarshal(w.Body.Bytes(), &detail); err != nil {
		t.Fatal(err)
	}
	if detail.Reading == nil || !strings.Contains(detail.Reading.Appendix, "108") || len(detail.Reading.Sources) < 2 {
		t.Fatal("incomplete reading detail")
	}
	// The API must reject a reading attached to a different model version.
	for i := range repo.library.Models {
		if repo.library.Models[i].Reading != nil {
			repo.library.Models[i].Reading.ModelVersion++
			break
		}
	}
	filename := filepath.Join(t.TempDir(), "mismatch.json")
	data, err := json.Marshal(repo.library)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filename, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = LoadRepository(filename); err == nil {
		t.Fatal("mismatched reading accepted")
	}
}
