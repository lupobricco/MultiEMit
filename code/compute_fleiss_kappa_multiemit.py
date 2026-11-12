"""
MultiEMiT Inter-Annotator Agreement (Fleiss' Kappa) Computation
================================================================
This script reconstructs the raw 3-annotator voting records for the MultiEMiT
benchmark by mapping source files from EMiT-main and ProSAT-main by URI.

It computes:
1. Binary Fleiss' Kappa across all individual categories for:
   - Textual partition (N = 2,023)
   - Multimodal partition with images (N = 645)
   - Multimodal paired control partition without images (N = 628)
2. Absolute gain (Delta kappa) and percentage increase on paired instances.
3. Summary metrics per pragmatic layer (Emotions, Irony, Target, Offensiveness, Semiotic Relation).

Outputs:
- datasets/annotations/fleiss_kappa_summary.csv
- datasets/annotations/fused_textual_annotations_3raters.csv
- datasets/annotations/fused_multimodal_annotations_3raters.csv
"""

import os
import pandas as pd
import numpy as np

def compute_fleiss_kappa(table):
    """
    Computes Fleiss' Kappa (Fleiss, 1971) for an (N, k) array where:
    - N is the number of subjects (posts),
    - k is the number of categories,
    - table[i, j] is the count of raters assigning subject i to category j.
    Each row must sum to n (the number of raters, here n=3).
    """
    table = np.asarray(table, dtype=np.float64)
    N, k = table.shape
    n = table[0].sum()
    
    p_j = table.sum(axis=0) / (N * n)
    P_i = (np.sum(table**2, axis=1) - n) / (n * (n - 1))
    
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j**2)
    
    if np.isclose(1.0, P_e):
        return 1.0
    
    kappa = (P_bar - P_e) / (1.0 - P_e)
    return kappa

def clean_tag(t):
    if not isinstance(t, str):
        return ''
    t = t.strip()
    mapping = {
        'GIoia': 'Gioia',
        'sorpresa': 'Sorpresa',
        'Contenituto': 'Contenuto',
        'Hate speech': 'Offensivita'
    }
    return mapping.get(t, t)

def main():
    print("=" * 70)
    print("MultiEMiT Inter-Annotator Agreement (Fleiss' Kappa) Pipeline")
    print("=" * 70)

    # 1. Load official golden datasets
    df_text_gold = pd.read_csv('datasets/splits/textual_full.csv')
    df_multi_gold = pd.read_csv('datasets/splits/multimodal_full.csv')
    
    text_uris = set(df_text_gold['uri'].astype(str))
    multi_uris = set(df_multi_gold['uri'].astype(str))
    print(f"Loaded Gold Standards: Textual N={len(text_uris)}, Multimodal N={len(multi_uris)}")

    # 2. Load anonymized source annotation files from datasets/annotations/raw/
    raw_dir = 'datasets/annotations/raw'
    f_rai_all = os.path.join(raw_dir, 'Commenti_Rai_annotation_all.csv')
    f_insta = os.path.join(raw_dir, 'Instagram_Rai_annotation_all.csv')
    f_twita1 = os.path.join(raw_dir, 'TWITA-annotation.csv')
    f_twita2_text = os.path.join(raw_dir, 'TWITA_2-annotation.csv')
    f_twita2_multi = os.path.join(raw_dir, 'TWITA_2_multi-annotation.csv')
    f_rai_sub = os.path.join(raw_dir, 'Commenti_Rai_annotation_subset.csv')

    df_rai_all = pd.read_csv(f_rai_all)
    df_insta = pd.read_csv(f_insta)
    df_twita1 = pd.read_csv(f_twita1)
    df_twita2_text = pd.read_csv(f_twita2_text)
    df_twita2_multi = pd.read_csv(f_twita2_multi)
    df_rai_sub = pd.read_csv(f_rai_sub)
    df_twita2_text = pd.read_csv(f_twita2_text)
    df_twita2_multi = pd.read_csv(f_twita2_multi)
    df_rai_sub = pd.read_csv(f_rai_sub)

    for df in [df_rai_all, df_insta, df_twita1, df_twita2_text, df_twita2_multi, df_rai_sub]:
        df['uri_str'] = df['uri'].astype(str)
        df['clean_tag'] = df['tag_name'].apply(clean_tag)

    # 3. Fuse Textual (N = 2,023)
    text_raw = pd.concat([
        df_rai_all[df_rai_all['uri_str'].isin(text_uris)],
        df_twita1[df_twita1['uri_str'].isin(text_uris)],
        df_twita2_text[df_twita2_text['uri_str'].isin(text_uris)]
    ], ignore_index=True)
    assert text_raw['uri_str'].nunique() == 2023, f"Expected 2023 URIs, got {text_raw['uri_str'].nunique()}"
    print(f"Textual Fused: {len(text_raw)} raw rows across {text_raw['uri_str'].nunique()} URIs (100% matched)")

    # 4. Fuse Multimodal With Images (N = 645)
    multi_raw = pd.concat([
        df_insta[df_insta['uri_str'].isin(multi_uris)],
        df_twita2_multi[df_twita2_multi['uri_str'].isin(multi_uris)]
    ], ignore_index=True)
    assert multi_raw['uri_str'].nunique() == 645, f"Expected 645 URIs, got {multi_raw['uri_str'].nunique()}"
    print(f"Multimodal Fused: {len(multi_raw)} raw rows across {multi_raw['uri_str'].nunique()} URIs (100% matched)")

    # 5. Fuse Multimodal Text-Only Control (Paired N = 628 with 3 raters)
    control_raw = pd.concat([
        df_rai_sub[df_rai_sub['uri_str'].isin(multi_uris)],
        df_twita2_text[df_twita2_text['uri_str'].isin(multi_uris)]
    ], ignore_index=True)
    print(f"Control Fused: {len(control_raw)} raw rows across {control_raw['uri_str'].nunique()} URIs")

    # Save fused datasets
    os.makedirs('datasets/annotations', exist_ok=True)
    text_raw.to_csv('datasets/annotations/fused_textual_annotations_3raters.csv', index=False)
    multi_raw.to_csv('datasets/annotations/fused_multimodal_annotations_3raters.csv', index=False)
    print("Saved fused annotation files in datasets/annotations/")

    # 6. Filter to exact 3-rater items
    raters_text = text_raw.groupby('uri_str')['annotator_id'].nunique()
    valid_text_uris = set(raters_text[raters_text == 3].index)

    raters_multi = multi_raw.groupby('uri_str')['annotator_id'].nunique()
    valid_multi_uris = set(raters_multi[raters_multi == 3].index)

    raters_ctrl = control_raw.groupby('uri_str')['annotator_id'].nunique()
    valid_ctrl_uris = set(raters_ctrl[raters_ctrl == 3].index)

    paired_uris = valid_multi_uris & valid_ctrl_uris
    print(f"Rater counts: Textual valid={len(valid_text_uris)}, Multimodal valid={len(valid_multi_uris)}, Paired={len(paired_uris)}")

    def get_kappas(raw_df, uris_to_use, tag_list):
        raw_df_filtered = raw_df[raw_df['uri_str'].isin(uris_to_use)]
        uri_tags = {}
        for uri, group in raw_df_filtered.groupby('uri_str'):
            uri_tags[uri] = group['clean_tag'].tolist()
        
        res = {}
        for tag in tag_list:
            table = []
            for uri in uris_to_use:
                labels = uri_tags.get(uri, [])
                yes = min(3, labels.count(tag))
                table.append([yes, 3 - yes])
            res[tag] = compute_fleiss_kappa(table)
        return res

    emotion_tags = ['Amore', 'Anticipazione', 'Disgusto', 'Fiducia', 'Gioia', 'Paura', 'Rabbia', 'Sorpresa', 'Tristezza', 'Neutro']
    irony_tags = ['Ironia', 'Sarcasmo']
    target_tags = ['Contenitore', 'Contenuto']
    offense_tags = ['Offensivita']
    relation_tags = ['Additiva', 'Divergente', 'Parallela']

    all_shared_tags = emotion_tags + irony_tags + target_tags + offense_tags

    k_text = get_kappas(text_raw, valid_text_uris, all_shared_tags)
    k_multi = get_kappas(multi_raw, valid_multi_uris, all_shared_tags + relation_tags)
    k_paired_img = get_kappas(multi_raw, paired_uris, all_shared_tags)
    k_paired_txt = get_kappas(control_raw, paired_uris, all_shared_tags)

    # Build summary dataframe
    rows = []
    category_names = {
        'Amore': ('Love', 'Emotion'),
        'Anticipazione': ('Anticipation', 'Emotion'),
        'Disgusto': ('Disgust', 'Emotion'),
        'Fiducia': ('Trust', 'Emotion'),
        'Gioia': ('Joy', 'Emotion'),
        'Paura': ('Fear', 'Emotion'),
        'Rabbia': ('Anger', 'Emotion'),
        'Sorpresa': ('Surprise', 'Emotion'),
        'Tristezza': ('Sadness', 'Emotion'),
        'Neutro': ('Neutral', 'Emotion'),
        'Ironia': ('Irony', 'Irony'),
        'Sarcasmo': ('Sarcasm', 'Irony'),
        'Contenitore': ('Container', 'Target'),
        'Contenuto': ('Content', 'Target'),
        'Offensivita': ('Offensive', 'Offensiveness'),
        'Additiva': ('Additive', 'Semiotic Relation'),
        'Divergente': ('Divergent', 'Semiotic Relation'),
        'Parallela': ('Parallel', 'Semiotic Relation'),
    }

    for tag in all_shared_tags:
        eng_name, layer = category_names[tag]
        k_t = k_text[tag]
        k_m = k_multi[tag]
        k_p_txt = k_paired_txt[tag]
        k_p_img = k_paired_img[tag]
        delta = k_p_img - k_p_txt
        pct = (delta / k_p_txt) * 100
        rows.append({
            'Category': eng_name,
            'Italian_Tag': tag,
            'Layer': layer,
            'Textual_N2023': round(k_t, 3),
            'Multimodal_N645': round(k_m, 3),
            'Paired_TextOnly_N628': round(k_p_txt, 3),
            'Paired_WithImage_N628': round(k_p_img, 3),
            'Delta_Gain': round(delta, 3),
            'Pct_Gain': round(pct, 2)
        })

    for tag in relation_tags:
        eng_name, layer = category_names[tag]
        k_m = k_multi[tag]
        rows.append({
            'Category': eng_name,
            'Italian_Tag': tag,
            'Layer': layer,
            'Textual_N2023': np.nan,
            'Multimodal_N645': round(k_m, 3),
            'Paired_TextOnly_N628': np.nan,
            'Paired_WithImage_N628': round(k_m, 3),
            'Delta_Gain': np.nan,
            'Pct_Gain': np.nan
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv('datasets/annotations/fleiss_kappa_summary.csv', index=False)
    print("Saved datasets/annotations/fleiss_kappa_summary.csv")

    print("\n" + "=" * 70)
    print("FLEISS KAPPA SUMMARY TABLE")
    print("=" * 70)
    print(df_summary.to_string(index=False))

if __name__ == '__main__':
    main()

