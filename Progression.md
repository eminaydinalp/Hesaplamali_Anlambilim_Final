# Proje İlerleme Takibi
## GSM8K_TR Üzerinde Seçici Fine-Tuning Araştırması

---

## FAZ 0 — Ortam Kurulumu

- [X] Python ortamı oluşturuldu (venv / conda)
- [X] Gerekli kütüphaneler yüklendi (`transformers`, `datasets`, `torch`, `openai`, `pandas`, `scikit-learn`)
- [X] GPU erişimi doğrulandı (CUDA / MPS / CPU fallback)
- [X] Proje klasör yapısı oluşturuldu:
  ```
  project/
  ├── data/
  ├── models/
  ├── results/
  ├── scripts/
  └── logs/
  ```
- [ ] Başarılı model API erişimi test edildi (GPT-4 / Gemini / Claude vb.)
- [X] Temel model (M1) seçildi ve erişimi doğrulandı

---

## FAZ 1 — Veri Hazırlığı

- [X] GSM8K_TR veri seti indirildi
- [X] Veri setinin toplam soru sayısı kontrol edildi
- [X] Veri formatı standartlaştırıldı (soru, cevap, çözüm adımları)
- [X] Test kümesi genel popülasyondan rastgele seçildi (≥ 500 soru)
- [X] Kalan sorular M1 değerlendirmesi için ayrıldı
- [X] Test kümesi ile kalan sorular arasında çakışma olmadığı doğrulandı
- [X] Test kümesi `data/test.jsonl` olarak kaydedildi
- [X] Numeric-only deney kapsamı için saat formatında final cevabı olan test örnekleri çıkarıldı / değiştirildi


---

## FAZ 2 — Baseline (M1) Değerlendirmesi ve Eğitim Kümesi Oluşturma

> Eğitim kümesi = M1'in yanlış cevapladığı sorular (≥ 500 soru)

- [X] M1 modeli GSM8K_TR soruları üzerinde çalıştırıldı
- [X] M1'in her soru için doğru / yanlış cevap verdiği kaydedildi → `data/m1_predictions.jsonl`
- [X] M1'in çözemediği sorular filtrelendi
- [X] Yanlış cevaplanan soru sayısının ≥ 500 olduğu doğrulandı (4085 soru)
- [X] Yanlış cevaplanan sorular eğitim kümesi olarak seçildi → `data/train_failed.jsonl`
- [X] M1 baseline başarı skoru hesaplandı ve kaydedildi → `logs/baseline.json`
- [X] Yanlış cevaplanan soruların doğrulanması için gpt-oss-120b modeline api isteği gönderilerek doğrulanmış eğitim kümesi elde edildi (2578 soru)
- [X] Doğrulanmış kümeden test kümesiyle çakışmayan nihai 500 eğitim sorusu seçildi → `data/train_final_500.jsonl`
- [X] Nihai eğitim kümesi ile test kümesi arasında id ve soru metni çakışması olmadığı doğrulandı → `logs/final_train_selection_summary.json`
- [X] Numeric-only deney kapsamı için saat formatında final cevabı olan eğitim örnekleri çıkarıldı / değiştirildi → `logs/numeric_only_repair_summary.json`

---

## FAZ 3 — Benzer Soru Üretimi

- [X] Başarılı model (Teacher Model) seçildi ve erişimi doğrulandı (`openai/gpt-oss-120b:free`)
- [X] Her başarısız soru (q1, q2, …) için benzer soru üretme scripti yazıldı (`data/train_final_500.jsonl` kaynak alınarak)
- [X] Benzer sorular (q11, q22, …) üretildi (500/500 geçerli)
- [X] Üretilen sorular gözden geçirildi (otomatik kalite kontrolü):
  - [X] Orijinal soru metniyle birebir aynı soru yok
  - [X] Orijinal final cevapla aynı final cevap yok
  - [X] Duplicate, saat-cevap, cevap/çözüm mismatch veya parse hatası yok
- [X] Benzer sorular `data/similar_questions.jsonl` olarak kaydedildi ve yeniden valide edildi → `logs/similar_questions_summary.json`
- [X] Teacher Model'den her başarısız soru için çözüm (r1, r2, …) alındı
- [X] Çözümler final eğitim kümesiyle hizalı olarak `data/solutions_final_500.jsonl` içinde kaydedildi

---

## FAZ 4 — Seçici Fine-Tuning Döngüsü

> Her soru için aşağıdaki döngü işletilir. Aktif model başlangıçta M1'dir.

- [X] Döngü scripti (`scripts/selective_loop.py`) yazıldı
- [X] Checkpoint kaydetme / yükleme mekanizması implement edildi
- [X] Her adım için log tutma mekanizması kuruldu (`logs/loop_log.csv`)
- [X] Döngü girdileri 500 hizalı örnekle dry-run üzerinden doğrulandı

**Döngü adımları (her qi için):**

> qi zaten Faz 2'de başarısız olduğu bilinen sorulardır, tekrar test edilmez.

- [X] ri (Teacher çözümü) ile aktif model fine-tune edildi → yeni model oluşturuldu
- [X] Yeni model ile qii (benzer soru) test edildi
- [X] Sonuç loglandı:
    - Başarılıysa → yeni model aktif model oldu ✓
    - Başarısızsa → önceki aktif model korundu ✓
- [X] Tüm başarısız sorular için döngü tamamlandı (500 adım; 405 kabul, 95 ret)
- [X] Her adımın sonucu `logs/loop_log.csv`'ye kaydedildi

---

## FAZ 5 — Kör Strateji (Karşılaştırma Grubu)

> Seçici strateji ile karşılaştırma için: her durumda M2 ile devam eden versiyon.

- [X] Kör strateji scripti (`scripts/blind_loop.py`) yazıldı
- [X] Aynı soru sırası ve aynı çözümler kullanıldı (adil karşılaştırma)
- [X] Döngü girdileri 500 hizalı örnekle dry-run üzerinden doğrulandı
- [X] Kör strateji tüm başarısız sorular için çalıştırıldı (500 adım; 389 benzer soru doğru, 111 yanlış)
- [X] Sonuçlar `logs/blind_loop_log.csv`'ye kaydedildi

---

## FAZ 6 — Test Kümesi Değerlendirmesi

- [X] Seçici strateji sonucundaki final modeli test kümesinde değerlendirildi
- [X] Kör strateji sonucundaki final modeli test kümesinde değerlendirildi
- [X] M1 (baseline) test kümesinde değerlendirildi (referans nokta)
- [X] Her model için aşağıdaki metrikler hesaplandı:
  - [X] Genel doğruluk (accuracy)
  - [X] Başarısız sorular üzerindeki doğruluk
  - [X] Test kümesi genelleştirme skoru
- [X] Sonuçlar `results/evaluation.csv` olarak kaydedildi

**Faz 6 test sonuçları:**

| Model | Doğru / 500 | Doğruluk | M1'in yanlış yaptığı 210 test sorusunda doğru | M1'e göre net fark |
|-------|-------------|----------|----------------------------------------------|--------------------|
| M1 baseline | 290 | 0.580 | 0 | 0 |
| Selective final adapter | 274 | 0.548 | 79 | -16 |
| Blind final adapter | 296 | 0.592 | 85 | +6 |

Not: Seçici strateji benzer soru testinde daha yüksek başarı göstermesine rağmen (`405/500 = 0.810`), sabit test kümesinde baseline'ın altında kaldı. Kör strateji benzer soru testinde daha düşük görünmesine rağmen (`389/500 = 0.778`), final test doğruluğunda en iyi sonucu verdi.

---

## FAZ 7 — Analiz ve Raporlama

- [X] Seçici strateji vs. kör strateji karşılaştırma tablosu oluşturuldu
- [X] Araştırma sorusu yanıtlandı:
  - [X] "Bir sorunun çözümünü öğrenmek, benzer soruları çözmeyi sağlıyor mu?"
  - [X] "Seçici model güncelleme, kör güncellemeden daha mı iyi?"
- [X] Öğrenme eğrisi grafiği çizildi (adım sayısı vs. başarı)
- [X] Başarılı ve başarısız fine-tuning adımları analiz edildi
- [X] Sonuçlar `results/final_report.md` olarak yazıldı
- [X] Görseller `results/figures/` klasörüne kaydedildi

**Faz 7 çıktıları:**

- `scripts/analyze_phase7.py`
- `results/final_report.md`
- `results/phase7_analysis.json`
- `results/figures/test_accuracy.svg`
- `results/figures/similar_learning_curve.svg`
- `results/figures/test_transitions.svg`
- `results/figures/test_accuracy.png`
- `results/figures/similar_learning_curve.png`
- `results/figures/test_transitions.png`

**Ana Faz 7 yorumu:** Teacher çözümüyle yapılan tek örneklik LoRA güncellemeleri lokal benzer soru başarısına çoğu zaman aktarım sağladı; ancak tek benzer soru üzerinden yapılan seçici kabul mekanizması final test genellemesini garanti etmedi. Bu deneyde final testte en iyi strateji blind oldu.

---

## Notlar

| Sembol | Anlam |
|--------|-------|
| `[ ]`  | Yapılmadı |
| `[X]`  | Tamamlandı |
| `[~]`  | Devam ediyor |
| `[!]`  | Sorun var, incelenmeli |

---

*Son güncelleme: 2026-06-10*
