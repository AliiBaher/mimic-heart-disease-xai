-- =============================================================================
-- create_heart_disease_dataset.sql
-- =============================================================================
-- Purpose : Build a machine-learning ready dataset for heart-disease risk
--           prediction from MIMIC-III tables.
--
-- Tables created (in order)
--   1. heart_base            – adult ICU cohort with demographics & LOS
--   2. heart_labels          – ICD-9 heart-disease flags per admission
--   3. vitals_first_day      – 24-hour ICU vital-sign averages
--   4. labs_first_day        – 24-hour lab-value averages
--   5. heart_disease_dataset – final merged ML table
--
-- ItemID reference (CareVue / Metavision)
--   Heart rate          : 211, 220045
--   Systolic BP         : 51, 442, 455, 220050, 220179
--   Diastolic BP        : 8368, 8440, 8441, 220051, 220180
--   Mean BP             : 52, 443, 456, 220052, 220181, 225312
--   SpO2                : 646, 220277
--   Respiratory rate    : 618, 220210, 224690
--   Temperature (°F→°C) : 678, 223761  |  direct °C: 676, 223762
--
-- Lab itemid reference
--   Creatinine : 50912  |  BUN        : 51006  |  Sodium     : 50983
--   Potassium  : 50971  |  Chloride   : 50902  |  Bicarbonate: 50882
--   Calcium    : 50893  |  Glucose    : 50931  |  Platelet   : 51265
--   WBC        : 51301  |  Hemoglobin : 51222  |  Hematocrit : 51221
--   Lactate    : 50813
-- =============================================================================


-- =============================================================================
-- 1. heart_base
--    Adult patients (age >= 18) with at least one ICU stay.
--    Age is capped at 91.4 to handle MIMIC-III privacy shift for >89-year-olds.
-- =============================================================================
DROP TABLE IF EXISTS heart_base;

CREATE TABLE heart_base AS
SELECT
    p.subject_id,
    a.hadm_id,
    i.icustay_id,
    p.gender,
    -- Cap age: MIMIC-III shifts DOB for patients older than 89, resulting in
    -- implausible values (~300 yrs). 91.4 is the MIMIC-III convention ceiling.
    LEAST(
        DATE_PART('year', AGE(a.admittime, p.dob)),
        91.4
    )::NUMERIC(5,1)                                 AS age,
    a.admittime,
    a.dischtime,
    i.intime                                         AS icu_intime,
    i.outtime                                        AS icu_outtime,
    ROUND(i.los::NUMERIC, 4)                         AS icu_los_days,
    ROUND(
        EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 86400.0
    , 4)                                             AS hosp_los_days,
    a.hospital_expire_flag,
    i.first_careunit,
    a.admission_type,
    a.ethnicity
FROM patients      p
JOIN admissions    a ON p.subject_id = a.subject_id
JOIN icustays      i ON a.hadm_id    = i.hadm_id
WHERE
    -- Adults only
    DATE_PART('year', AGE(a.admittime, p.dob)) >= 18
    -- Exclude rows where ICU outtime is missing
    AND i.outtime IS NOT NULL
    AND a.dischtime IS NOT NULL;

-- Keep only the first ICU stay per admission to avoid duplicate rows
DELETE FROM heart_base hb
WHERE EXISTS (
    SELECT 1
    FROM heart_base hb2
    WHERE hb2.hadm_id    = hb.hadm_id
      AND hb2.icustay_id < hb.icustay_id
);

SELECT 'heart_base rows:' AS check_label, COUNT(*) AS n FROM heart_base;


-- =============================================================================
-- 2. heart_labels
--    ICD-9 heart-disease flags per admission:
--      ischemic_hd  : codes 410–414 (ischaemic heart disease / CAD)
--      dysrhythmia  : code  427     (cardiac dysrhythmias)
--      heart_failure: code  428     (heart failure)
--      heart_disease: 1 if any of the above is present
-- =============================================================================
DROP TABLE IF EXISTS heart_labels;

CREATE TABLE heart_labels AS
SELECT
    hadm_id,
    MAX(CASE
        WHEN LEFT(icd9_code, 3) BETWEEN '410' AND '414' THEN 1
        ELSE 0
    END)::SMALLINT                                   AS ischemic_hd,
    MAX(CASE
        WHEN LEFT(icd9_code, 3) = '427' THEN 1
        ELSE 0
    END)::SMALLINT                                   AS dysrhythmia,
    MAX(CASE
        WHEN LEFT(icd9_code, 3) = '428' THEN 1
        ELSE 0
    END)::SMALLINT                                   AS heart_failure,
    MAX(CASE
        WHEN LEFT(icd9_code, 3) BETWEEN '410' AND '414'
          OR LEFT(icd9_code, 3) = '427'
          OR LEFT(icd9_code, 3) = '428' THEN 1
        ELSE 0
    END)::SMALLINT                                   AS heart_disease
FROM diagnoses_icd
GROUP BY hadm_id;

SELECT 'heart_labels rows:' AS check_label, COUNT(*) AS n FROM heart_labels;


-- =============================================================================
-- 3. vitals_first_day
--    Average vital signs recorded in chartevents during the first 24 hours
--    of the ICU stay. Only plausible physiological values are accepted.
--    Temperature is normalised to Celsius regardless of source unit.
-- =============================================================================
DROP TABLE IF EXISTS vitals_first_day;

CREATE TABLE vitals_first_day AS
WITH raw_vitals AS (
    SELECT
        c.icustay_id,
        c.itemid,
        -- Normalise temperature to Celsius
        CASE
            WHEN c.itemid IN (678, 223761)           -- Fahrenheit sources
            THEN (c.valuenum - 32.0) * 5.0 / 9.0
            ELSE c.valuenum
        END AS valuenum
    FROM chartevents c
    JOIN heart_base  b ON c.icustay_id = b.icustay_id
    WHERE
        c.itemid IN (
            -- Heart rate
            211, 220045,
            -- Systolic BP
            51, 442, 455, 220050, 220179,
            -- Diastolic BP
            8368, 8440, 8441, 220051, 220180,
            -- Mean BP
            52, 443, 456, 220052, 220181, 225312,
            -- SpO2
            646, 220277,
            -- Respiratory rate
            618, 220210, 224690,
            -- Temperature: Fahrenheit sources converted above, direct Celsius also included
            676, 678, 223761, 223762
        )
        AND c.valuenum IS NOT NULL
        AND c.valuenum > 0
        -- First 24 hours of ICU admission only
        AND c.charttime >= b.icu_intime
        AND c.charttime <= b.icu_intime + INTERVAL '24 hours'
        -- Exclude erroneous / manually stopped entries
        AND (c.error IS NULL OR c.error = 0)
)
SELECT
    icustay_id,

    -- Heart rate (20–300 bpm)
    ROUND(AVG(CASE WHEN itemid IN (211, 220045)
                    AND valuenum BETWEEN 20  AND 300
              THEN valuenum END)::NUMERIC, 2)        AS heart_rate,

    -- Systolic BP (40–300 mmHg)
    ROUND(AVG(CASE WHEN itemid IN (51, 442, 455, 220050, 220179)
                    AND valuenum BETWEEN 40  AND 300
              THEN valuenum END)::NUMERIC, 2)        AS sbp,

    -- Diastolic BP (20–200 mmHg)
    ROUND(AVG(CASE WHEN itemid IN (8368, 8440, 8441, 220051, 220180)
                    AND valuenum BETWEEN 20  AND 200
              THEN valuenum END)::NUMERIC, 2)        AS dbp,

    -- Mean BP (20–250 mmHg)
    ROUND(AVG(CASE WHEN itemid IN (52, 443, 456, 220052, 220181, 225312)
                    AND valuenum BETWEEN 20  AND 250
              THEN valuenum END)::NUMERIC, 2)        AS mbp,

    -- SpO2 (50–100 %)
    ROUND(AVG(CASE WHEN itemid IN (646, 220277)
                    AND valuenum BETWEEN 50  AND 100
              THEN valuenum END)::NUMERIC, 2)        AS spo2,

    -- Respiratory rate (4–60 breaths/min)
    ROUND(AVG(CASE WHEN itemid IN (618, 220210, 224690)
                    AND valuenum BETWEEN 4   AND 60
              THEN valuenum END)::NUMERIC, 2)        AS resp_rate,

    -- Temperature in °C (25–45 °C after conversion)
    ROUND(AVG(CASE WHEN itemid IN (676, 678, 223761, 223762)
                    AND valuenum BETWEEN 25  AND 45
              THEN valuenum END)::NUMERIC, 2)        AS temp_celsius

FROM raw_vitals
GROUP BY icustay_id;

SELECT 'vitals_first_day rows:' AS check_label, COUNT(*) AS n FROM vitals_first_day;


-- =============================================================================
-- 4. labs_first_day
--    Average lab values from labevents during the first 24 hours of the
--    ICU stay. Physiological clipping prevents outlier contamination.
-- =============================================================================
DROP TABLE IF EXISTS labs_first_day;

CREATE TABLE labs_first_day AS
WITH raw_labs AS (
    SELECT
        le.hadm_id,
        le.itemid,
        le.valuenum
    FROM labevents  le
    JOIN heart_base b ON le.hadm_id = b.hadm_id
    WHERE
        le.itemid IN (
            50912,          -- Creatinine
            51006,          -- Urea Nitrogen (BUN)
            50983,          -- Sodium
            50971,          -- Potassium
            50902,          -- Chloride
            50882,          -- Bicarbonate
            50893,          -- Calcium, Total
            50931,          -- Glucose (blood)
            51265,          -- Platelet Count
            51301,          -- White Blood Cells (WBC)
            51222,          -- Hemoglobin
            51221,          -- Hematocrit
            50813           -- Lactate
        )
        AND le.valuenum IS NOT NULL
        AND le.valuenum > 0
        -- First 24 hours of ICU admission
        AND le.charttime >= b.icu_intime
        AND le.charttime <= b.icu_intime + INTERVAL '24 hours'
)
SELECT
    hadm_id,

    ROUND(AVG(CASE WHEN itemid = 50912
                    AND valuenum BETWEEN 0.1  AND 30    THEN valuenum END)::NUMERIC, 3) AS creatinine,
    ROUND(AVG(CASE WHEN itemid = 51006
                    AND valuenum BETWEEN 1    AND 300   THEN valuenum END)::NUMERIC, 2) AS bun,
    ROUND(AVG(CASE WHEN itemid = 50983
                    AND valuenum BETWEEN 100  AND 180   THEN valuenum END)::NUMERIC, 2) AS sodium,
    ROUND(AVG(CASE WHEN itemid = 50971
                    AND valuenum BETWEEN 1.5  AND 10    THEN valuenum END)::NUMERIC, 3) AS potassium,
    ROUND(AVG(CASE WHEN itemid = 50902
                    AND valuenum BETWEEN 70   AND 130   THEN valuenum END)::NUMERIC, 2) AS chloride,
    ROUND(AVG(CASE WHEN itemid = 50882
                    AND valuenum BETWEEN 5    AND 50    THEN valuenum END)::NUMERIC, 2) AS bicarbonate,
    ROUND(AVG(CASE WHEN itemid = 50893
                    AND valuenum BETWEEN 1    AND 20    THEN valuenum END)::NUMERIC, 3) AS calcium_total,
    ROUND(AVG(CASE WHEN itemid = 50931
                    AND valuenum BETWEEN 30   AND 1500  THEN valuenum END)::NUMERIC, 2) AS glucose,
    ROUND(AVG(CASE WHEN itemid = 51265
                    AND valuenum BETWEEN 10   AND 1500  THEN valuenum END)::NUMERIC, 2) AS platelet,
    ROUND(AVG(CASE WHEN itemid = 51301
                    AND valuenum BETWEEN 0.5  AND 500   THEN valuenum END)::NUMERIC, 3) AS wbc,
    ROUND(AVG(CASE WHEN itemid = 51222
                    AND valuenum BETWEEN 2    AND 25    THEN valuenum END)::NUMERIC, 3) AS hemoglobin,
    ROUND(AVG(CASE WHEN itemid = 51221
                    AND valuenum BETWEEN 5    AND 75    THEN valuenum END)::NUMERIC, 2) AS hematocrit,
    ROUND(AVG(CASE WHEN itemid = 50813
                    AND valuenum BETWEEN 0.1  AND 30    THEN valuenum END)::NUMERIC, 3) AS lactate

FROM raw_labs
GROUP BY hadm_id;

SELECT 'labs_first_day rows:' AS check_label, COUNT(*) AS n FROM labs_first_day;


-- =============================================================================
-- 5. heart_disease_dataset
--    Final ML-ready table: one row per ICU stay with all features and labels.
-- =============================================================================
DROP TABLE IF EXISTS heart_disease_dataset;

CREATE TABLE heart_disease_dataset AS
SELECT
    -- ── Identifiers ───────────────────────────────────────────────
    b.subject_id,
    b.hadm_id,
    b.icustay_id,

    -- ── Demographics ──────────────────────────────────────────────
    b.gender,
    b.age,
    b.ethnicity,
    b.admission_type,
    b.first_careunit,

    -- ── Stay characteristics ──────────────────────────────────────
    b.icu_los_days,
    b.hosp_los_days,

    -- ── Vital signs (first 24 h ICU) ──────────────────────────────
    v.heart_rate,
    v.sbp,
    v.dbp,
    v.mbp,
    v.spo2,
    v.resp_rate,
    v.temp_celsius,

    -- ── Lab values (first 24 h ICU) ───────────────────────────────
    l.creatinine,
    l.bun,
    l.sodium,
    l.potassium,
    l.chloride,
    l.bicarbonate,
    l.calcium_total,
    l.glucose,
    l.platelet,
    l.wbc,
    l.hemoglobin,
    l.hematocrit,
    l.lactate,

    -- ── Diagnostic labels ─────────────────────────────────────────
    COALESCE(hl.ischemic_hd,   0)::SMALLINT         AS ischemic_hd,
    COALESCE(hl.dysrhythmia,   0)::SMALLINT         AS dysrhythmia,
    COALESCE(hl.heart_failure, 0)::SMALLINT         AS heart_failure,
    COALESCE(hl.heart_disease, 0)::SMALLINT         AS heart_disease,

    -- ── Outcome label ─────────────────────────────────────────────
    b.hospital_expire_flag

FROM heart_base b
LEFT JOIN vitals_first_day  v  ON b.icustay_id = v.icustay_id
LEFT JOIN labs_first_day    l  ON b.hadm_id    = l.hadm_id
LEFT JOIN heart_labels      hl ON b.hadm_id    = hl.hadm_id;


-- =============================================================================
-- Final quality checks
-- =============================================================================

-- Total row count
SELECT 'heart_disease_dataset rows:' AS check_label, COUNT(*) AS n
FROM heart_disease_dataset;

-- Heart-disease class distribution
SELECT
    'heart_disease class distribution' AS check_label,
    heart_disease,
    COUNT(*)                            AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM heart_disease_dataset
GROUP BY heart_disease
ORDER BY heart_disease;

-- Sub-label distribution
SELECT
    'sub-label distribution' AS check_label,
    SUM(ischemic_hd)   AS ischemic_hd_n,
    SUM(dysrhythmia)   AS dysrhythmia_n,
    SUM(heart_failure) AS heart_failure_n,
    SUM(heart_disease) AS any_heart_disease_n
FROM heart_disease_dataset;

-- Mortality by heart-disease label
SELECT
    'mortality by heart_disease label' AS check_label,
    heart_disease,
    SUM(hospital_expire_flag)           AS deaths,
    COUNT(*)                            AS total,
    ROUND(SUM(hospital_expire_flag) * 100.0 / COUNT(*), 2) AS mortality_pct
FROM heart_disease_dataset
GROUP BY heart_disease
ORDER BY heart_disease;

-- Missing-value counts per feature column
SELECT
    'missing values' AS check_label,
    COUNT(*) FILTER (WHERE heart_rate   IS NULL) AS heart_rate_null,
    COUNT(*) FILTER (WHERE sbp          IS NULL) AS sbp_null,
    COUNT(*) FILTER (WHERE dbp          IS NULL) AS dbp_null,
    COUNT(*) FILTER (WHERE mbp          IS NULL) AS mbp_null,
    COUNT(*) FILTER (WHERE spo2         IS NULL) AS spo2_null,
    COUNT(*) FILTER (WHERE resp_rate    IS NULL) AS resp_rate_null,
    COUNT(*) FILTER (WHERE temp_celsius IS NULL) AS temp_celsius_null,
    COUNT(*) FILTER (WHERE creatinine   IS NULL) AS creatinine_null,
    COUNT(*) FILTER (WHERE bun          IS NULL) AS bun_null,
    COUNT(*) FILTER (WHERE sodium       IS NULL) AS sodium_null,
    COUNT(*) FILTER (WHERE lactate      IS NULL) AS lactate_null
FROM heart_disease_dataset;
