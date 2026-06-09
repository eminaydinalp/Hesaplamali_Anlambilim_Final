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
- [ ] Test kümesi ile kalan sorular arasında çakışma olmadığı doğrulandı
- [X] Test kümesi `data/test.json` olarak kaydedildi


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

---

## FAZ 3 — Benzer Soru Üretimi

- [ ] Başarılı model (Teacher Model) seçildi ve erişimi doğrulandı
- [X] Her başarısız soru (q1, q2, …) için benzer soru üretme scripti yazıldı (`data/train_final_500.jsonl` kaynak alınarak)
- [ ] Benzer sorular (q11, q22, …) üretildi
- [ ] Üretilen sorular gözden geçirildi (kalite kontrolü):
  - [ ] Konusu orijinal soruyla aynı mı?
  - [ ] Sayılar / isimler değiştirilmiş mi?
  - [ ] Zorluk seviyesi benzer mi?
- [ ] Benzer sorular `data/similar_questions.jsonl` olarak kaydedildi
- [ ] Teacher Model'den her başarısız soru için çözüm (r1, r2, …) alındı
- [ ] Çözümler `data/solutions.jsonl` olarak kaydedildi

---

## FAZ 4 — Seçici Fine-Tuning Döngüsü

> Her soru için aşağıdaki döngü işletilir. Aktif model başlangıçta M1'dir.

- [X] Döngü scripti (`scripts/selective_loop.py`) yazıldı
- [X] Checkpoint kaydetme / yükleme mekanizması implement edildi
- [X] Her adım için log tutma mekanizması kuruldu (`logs/loop_log.csv`)

**Döngü adımları (her qi için):**

> qi zaten Faz 2'de başarısız olduğu bilinen sorulardır, tekrar test edilmez.

- [ ] ri (Teacher çözümü) ile aktif model fine-tune edildi → yeni model oluşturuldu
- [ ] Yeni model ile qii (benzer soru) test edildi
- [ ] Sonuç loglandı:
    - Başarılıysa → yeni model aktif model oldu ✓
    - Başarısızsa → önceki aktif model korundu ✓
- [ ] Tüm başarısız sorular için döngü tamamlandı
- [ ] Her adımın sonucu `logs/loop_log.csv`'ye kaydedildi

---

## FAZ 5 — Kör Strateji (Karşılaştırma Grubu)

> Seçici strateji ile karşılaştırma için: her durumda M2 ile devam eden versiyon.

- [ ] Kör strateji scripti (`scripts/blind_loop.py`) yazıldı
- [ ] Aynı soru sırası ve aynı çözümler kullanıldı (adil karşılaştırma)
- [ ] Kör strateji tüm başarısız sorular için çalıştırıldı
- [ ] Sonuçlar `logs/blind_loop_log.csv`'ye kaydedildi

---

## FAZ 6 — Test Kümesi Değerlendirmesi

- [ ] Seçici strateji sonucundaki final modeli test kümesinde değerlendirildi
- [ ] Kör strateji sonucundaki final modeli test kümesinde değerlendirildi
- [ ] M1 (baseline) test kümesinde değerlendirildi (referans nokta)
- [ ] Her model için aşağıdaki metrikler hesaplandı:
  - [ ] Genel doğruluk (accuracy)
  - [ ] Başarısız sorular üzerindeki doğruluk
  - [ ] Test kümesi genelleştirme skoru
- [ ] Sonuçlar `results/evaluation.csv` olarak kaydedildi

---

## FAZ 7 — Analiz ve Raporlama

- [ ] Seçici strateji vs. kör strateji karşılaştırma tablosu oluşturuldu
- [ ] Araştırma sorusu yanıtlandı:
  - [ ] "Bir sorunun çözümünü öğrenmek, benzer soruları çözmeyi sağlıyor mu?"
  - [ ] "Seçici model güncelleme, kör güncellemeden daha mı iyi?"
- [ ] Öğrenme eğrisi grafiği çizildi (adım sayısı vs. başarı)
- [ ] Başarılı ve başarısız fine-tuning adımları analiz edildi
- [ ] Sonuçlar `results/final_report.md` olarak yazıldı
- [ ] Görseller `results/figures/` klasörüne kaydedildi

---

## Notlar

| Sembol | Anlam |
|--------|-------|
| `[ ]`  | Yapılmadı |
| `[X]`  | Tamamlandı |
| `[~]`  | Devam ediyor |
| `[!]`  | Sorun var, incelenmeli |

---

*Son güncelleme: 2026-06-05*
