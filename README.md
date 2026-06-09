# GSM8K_TR Seçici Fine-Tuning Deneyi

Bu proje, GSM8K_TR üzerinde bir modelin çözemediği soruların öğretmen model çözümleriyle fine-tune edilmesinin benzer sorulara genelleme sağlayıp sağlamadığını incelemek için hazırlanmıştır.

Ana karşılaştırma iki strateji arasındadır:

- Seçici strateji: Model, yeni öğrendiği çözümle benzer soruyu çözebiliyorsa güncellenmiş modelle devam eder.
- Kör strateji: Her fine-tuning adımından sonra son modelle devam eder.

## Klasör Yapısı

```text
.
├── data/
├── models/
├── results/
├── scripts/
└── logs/
```

## Ortam Kurulumu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cihaz Kontrolü

CUDA, MPS veya CPU fallback durumunu kontrol etmek için:

```bash
source .venv/bin/activate
python scripts/check_device.py
```

Script çıktıyı ekrana basar ve aynı bilgiyi `logs/device_check.json` dosyasına yazar.

## Temel Model Kontrolü

M1 için varsayılan model `Qwen/Qwen3.5-4B` olarak seçilmiştir. Model daha önce Hugging Face cache'ine indirilmişse tekrar indirilmeden kullanılabilir:

```bash
source .venv/bin/activate
python scripts/verify_base_model.py
```

Script `local_files_only=True` ile çalışır; yani model cache'te yoksa internete çıkıp indirme yapmaz. Çıktıyı `logs/base_model_check.json` dosyasına yazar.

## GSM8K_TR Veri Seti

Veri setini Hugging Face üzerinden yükleyip proje formatına çevirmek için:

```bash
source .venv/bin/activate
python scripts/download_gsm8k_tr.py
```

Script `ytu-ce-cosmos/gsm8k_tr` veri setini okur, standart JSONL çıktısını `data/gsm8k_tr.jsonl` dosyasına yazar ve özet bilgiyi `logs/gsm8k_tr_dataset_info.json` içinde saklar.

## M1 Başarısız Soru Kümesi

Önce veri setindeki çözüm metinlerinden referans final cevapları çıkarın:

```bash
source .venv/bin/activate
python scripts/prepare_reference_answers.py
```

Ardından M1'i tüm sorular üzerinde çalıştırıp yanlış cevaplananları eğitim kümesine ayırın:

```bash
python scripts/evaluate_m1_failures.py --batch-size 4
```

Çıktılar:

- `data/gsm8k_tr_references.jsonl`: `id`, `reference_answer_raw`, `reference_answer`
- `data/m1_predictions.jsonl`: M1'in tüm yanıtları ve doğru/yanlış bilgisi
- `data/train_failed.jsonl`: M1'in yanlış çözdüğü sorular
- `logs/m1_eval_summary.json`: değerlendirme özeti

Komut resume desteklidir; yarıda kesilirse aynı komutu tekrar çalıştırmak kaldığı yerden devam eder. Baştan başlatmak için `--restart` kullanın.

## Qwen Test Baseline

Eğitimden önce M1/Qwen modelinin sabit test kümesindeki performansını ölçmek için:

```bash
source .venv/bin/activate
python scripts/evaluate_qwen_test_baseline.py --batch-size 4 --max-new-tokens 256
```

Girdi dosyaları:

- `data/test.jsonl`
- `data/test_with_reference_numeric.jsonl`

Çıktılar:

- `data/test_qwen_predictions.jsonl`
- `data/test_qwen_failed.jsonl`
- `logs/qwen_test_baseline_summary.json`

Komut resume desteklidir; yarıda kesilirse aynı komutla kaldığı yerden devam eder. Baştan başlatmak için `--restart` kullanın.

## OpenRouter Teacher Doğrulaması

OpenRouter API anahtarını `.env` dosyasına yazın:

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-oss-120b:free
```

M1'in yanlış işaretlenen cevaplarını `gpt-oss-120b` ile tekrar değerlendirmek için:

```bash
source .venv/bin/activate
python scripts/evaluate_openrouter_teacher.py
```

Bu script `data/train_failed.jsonl` içindeki sorulara M1 ile aynı promptu gönderir. Teacher cevabı referans cevapla uyuşuyor ve M1 cevabı teacher cevabıyla uyuşmuyorsa soru doğrulanmış yanlış kabul edilir.

Çıktılar:

- `data/gpt_oss_120b_predictions.jsonl`: teacher model yanıtları
- `data/train_failed_verified.jsonl`: doğrulanmış yanlış cevaplar
- `data/train_failed_disputed.jsonl`: teacher/reference uyuşmayan veya kontrol dışı kalan örnekler
- `data/solutions.jsonl`: doğrulanmış yanlışlar için teacher çözümü
- `logs/openrouter_teacher_summary.json`: özet

Komut resume desteklidir; yarıda kesilirse aynı komutla kaldığı yerden devam eder. Baştan başlatmak için `--restart` kullanın.

## Nihai Eğitim Kümesi

Doğrulanmış başarısız sorulardan test kümesiyle çakışmayan 500 örnek seçmek için:

```bash
source .venv/bin/activate
python scripts/select_final_training_set.py --restart
```

Script `data/test.jsonl` içindeki `source_id` alanlarını `train-<source_id>` biçimine çevirerek `data/train_failed_verified.jsonl` id'leriyle karşılaştırır. Ayrıca normalize edilmiş soru metinleriyle ikinci bir çakışma kontrolü yapar.

Çıktılar:

- `data/train_final_500.jsonl`: nihai 500 eğitim sorusu
- `data/solutions_final_500.jsonl`: aynı 500 soru için teacher çözümleri
- `logs/final_train_selection_summary.json`: seçim ve overlap özeti

## Benzer Soru Üretimi

Nihai eğitim kümesi seçildikten sonra bu 500 soru için benzer soru üretmek için:

```bash
source .venv/bin/activate
python scripts/generate_similar_questions.py
```

Script varsayılan olarak `data/train_final_500.jsonl` dosyasını okur ve her satır için aynı matematiksel yapıda yeni bir Türkçe soru, teacher çözümü ve numeric final cevap üretir.

Çıktılar:

- `data/similar_questions.jsonl`: benzer sorular, çözümler ve numeric cevaplar
- `logs/similar_questions_summary.json`: özet

Önce küçük test için:

```bash
python scripts/generate_similar_questions.py --limit 5
```

Komut resume desteklidir; yarıda kesilirse aynı komutla kaldığı yerden devam eder. Baştan başlatmak için `--restart` kullanın.

## Seçici LoRA Fine-Tuning Döngüsü

Faz 4 döngüsü için:

```bash
source .venv/bin/activate
python scripts/selective_loop.py
```

Script şu girdileri hizalar:

- `data/train_final_500.jsonl`
- `data/solutions_final_500.jsonl`
- `data/similar_questions.jsonl`

Varsayılan gerçek deney koşusunda en az 500 hizalanmış örnek beklenir. Faz 3 tümden bitmeden, `similar_questions.jsonl` içinde birkaç hizalanmış örnek oluştuktan sonra küçük prova yapmak için:

```bash
python scripts/selective_loop.py --limit 5 --allow-incomplete --dry-run
```

LoRA stratejisinde base model tekrar tekrar kaydedilmez. Diskte yalnızca seçici döngünün son kabul edilmiş adapter'ı ve geçici aday adapter tutulur:

- `models/selective_loop/active_adapter`
- `models/selective_loop/candidate_adapter_tmp`

Her adım `logs/loop_log.csv` dosyasına yazılır. Resume bilgisi `logs/selective_loop_state.json`, özet bilgi `logs/selective_loop_summary.json` içindedir. Baştan başlatmak için `--restart` kullanın.

## Proje Fazları

Ayrıntılı iş takibi `Progression.md` dosyasındadır. Faz 0 ortam kurulumu ve temel klasör yapısını kapsar.
