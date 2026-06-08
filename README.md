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

## Proje Fazları

Ayrıntılı iş takibi `Progression.md` dosyasındadır. Faz 0 ortam kurulumu ve temel klasör yapısını kapsar.
