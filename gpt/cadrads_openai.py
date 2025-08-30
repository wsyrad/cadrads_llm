import openai
import os
import pandas as pd
import time
import re
import json

openai.api_key =  # api key

def categorize(report_text, model):
    prompt = f"""
    @REPORT
    ```
    {report_text}
    ```

    <Objective>
    @REPORT is a radiology report of coronary CT.
    You are a cardiac radiologist with 10 years of experience, interpreting 500 cases of coronary CT each year using the CAD-RADS 2.0.
    Select the most appropriate CAD-RADS category according to the <Task details> and <CAD-RADS 2.0 guidelines> below.
    !!Do not explain the process. Respond strictly in JSON format following the <Output format>.

    <Task details>
    1. “r/o” OR “suspected” OR “suspicious for” a feature indicates the presence of that feature.
    2. Refer to the abbreviations below.
        • p = proximal
        • m = mid
        • d = distal
        • LM = left main coronary artery
        • LAD = left anterior descending artery
        • LCx = left circumflex artery
        • RCA = right coronary artery
        • LCA = left coronary artery
        • Dg = diagonal branch
        • RI = ramus intermedius
        • OM = obtuse marginal branch
        • rPD = right posterior descending artery
        • rPL = right posterolateral branch
        • ISR = in-stent restenosis
        Example:
        “pRCA” = “proximal right coronary artery”
        “m-dLAD” = “mid left anterior descending artery” AND “distal left anterior descending artery”

    <CAD-RADS 2.0 guidelines>
    1. Category for stenosis severity [0, 1, 2, 3, 4A, 4B, 5, N]
    !!Always select the highest category among the possible categories.
        • 5: “Degree of maximal coronary stenosis = 100% (Total or subtotal occlusion)”
        • 4B: “Degree of LM stenosis ≥ 50% but < 100% (Moderate or severe stenosis in LM)” OR “Degree of maximal coronary stenosis ≥ 70% but < 100% (Severe stenosis) in all three coronary vessels (LAD, LCx, RCA)”
        • 4A: “Degree of maximal coronary stenosis ≥ 70% but < 100% (Severe stenosis) in one or two coronary vessels”
        • 3: “Degree of maximal coronary stenosis ≥ 50% but < 70% (Moderate stenosis)”
        • 2: “Degree of maximal coronary stenosis ≥ 25% but < 50% (Mild stenosis)”
        • 1: “Degree of maximal coronary stenosis > 0% but < 25% (Minimal stenosis)”
        • 0: “Degree of maximal coronary stenosis = 0% (No visible stenosis)”
    !!EXCEPTION1: If there is at least one coronary segment that is non-diagnostic or limited in evaluation, AND degree of maximal coronary stenosis is < 50%, select “N”.
    !!EXCEPTION2: When a bypass graft is present, exclude the stenosis of the coronary segment bypassed by the graft and evaluate the graft and the distal coronary segment instead.

    2. Category for plaque burden [No, P1, P2, P3, P4]
    STEP1: Obtain the Coronary Artery Calcium Score.
    STEP2: Calculate the Segment Involvement Score.
        • Count the number of coronary artery segments with plaque among the 16 segments below. 
            • 16 segments: LM, pLAD, mLAD, dLAD, 1st Dg, 2nd Dg, RI, pLCx, dLCx, 1st OM, 2nd OM, pRCA, mRCA, dRCA, rPD, rPL
        • !!If segments are described as a range, count each segment in the range. For example, p-dLAD indicates pLAD, mLAD, dLAD, and should be counted as 3 segments.
    STEP3: Select the category for plaque burden.
        • If the Coronary Artery Calcium Score is unknown, use the Segment Involvement Score.
        • If there is at least one stent anywhere in the coronary system, use the Segment Involvement Score.
        • If the Coronary Artery Calcium Score is zero, but coronary plaque exists, use the Segment Involvement Score.
        • In all other cases, use the Coronary Artery Calcium Score.
        1) Segment Involvement Score classification:
            • No: 0 segments
            • P1: 1 or 2 segments
            • P2: 3 or 4 segments
            • P3: 5, 6, or 7 segments
            • P4: ≥ 8 segments
        2) Coronary Artery Calcium Score classification:
            • No: Score = 0
            • P1: Score > 0 but ≤ 100
            • P2: Score > 100 but ≤ 300
            • P3: Score > 300 but ≤ 1000
            • P4: Score ≥ 1000

    3. Modifier N (non-diagnostic) [N, No]
        • N: If there is at least one coronary segment that is non-diagnostic or limited in evaluation, AND degree of maximal coronary stenosis is ≥ 50%, select “N”.
        • No: In all other cases

    4. Modifier HRP (high-risk plaque) [HRP, No]
        • HRP: “Plaque with two or more of the following characteristics: positive remodeling, low-attenuation, spotty calcification, napkin-ring sign” OR “presence of high-risk plaque” OR “presence of vulnerable plaque”
        • No: In all other cases

    5. Modifier S (stent) [S, No]
        • S: Presence of at least one stent anywhere in the coronary system
        • No: In all other cases

    6. Modifier G (graft) [G, No]
        • G: Presence of at least one coronary artery bypass graft
        • No: In all other cases

    7. Modifier E (exceptions) [E, No]
        • E: Presence of non-atherosclerotic causes of coronary abnormalities (coronary dissection, anomalous origin of major coronary artery, coronary artery aneurysm or pseudo-aneurysm, coronary vasculitis, coronary artery fistula, extrinsic coronary artery compression, or arteriovenous malformation)
        • No: In all other cases

   !!Before providing the output, double-check if categories and modifiers are appropriately selected according to the <CAD-RADS 2.0 Guidelines> and <Task Details>.

    <Output format>
    {{
    "Category for stenosis severity": "string(0, 1, 2, 3, 4A, 4B, 5, N)",
    "Category for plaque burden": "string(No, P1, P2, P3, P4)",
    "Modifier N": "string(No, N)",
    "Modifier HRP": "string(No, HRP)",
    "Modifier S": "string(No, S)",
    "Modifier G": "string(No, G)",
    "Modifier E": "string(No, E)"
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=1,
            temperature=0,  # for GPT-3.5-turbo, GPT-4, GPT-4o
            reasoning_effort="high"  # for O1
        )

        output = response.choices[0]['message']['content'].strip()
        cleaned_content = re.sub(r'```json|```', '', output).strip()
        cad_rads_dict = json.loads(cleaned_content)
        return cad_rads_dict
        
    except Exception as e:
        print(f"Error in generating CAD-RADS category: {e}")
        return{
            "Category for stenosis severity": "",
            "Category for plaque burden": "",
            "Modifier N": "",
            "Modifier HRP": "",
            "Modifier S": "",
            "Modifier G": "",
            "Modifier E": ""
        }

def process_reports(input_file, output_file):

    df = pd.read_excel(input_file)

    ids = df['id']
    reports = df['report']

    results = []

    for i in range(len(reports)):
        report_a = reports[i]
        retries = 5
        success = False 

        for attempt in range(retries):
            cad_rads_category = categorize(report_a, 'model')  # Enter the model to be used
            
            if all(value != "" for value in [
                cad_rads_category.get('Category for stenosis severity', ''),
                cad_rads_category.get('Category for plaque burden', ''),
                cad_rads_category.get('Modifier N', ''),
                cad_rads_category.get('Modifier HRP', ''),
                cad_rads_category.get('Modifier S', ''),
                cad_rads_category.get('Modifier G', ''),
                cad_rads_category.get('Modifier E', '')
            ]):
                success = True
                break
            else:
                print(f"Attempt {attempt + 1} failed for report {ids[i]}. Retrying...")
                time.sleep(30)

        if not success:
            print(f"Failed to obtain valid CAD-RADS results for report {ids[i]} after {retries} attempts.")

        modifiers = [
            cad_rads_category.get('Category for stenosis severity', ''),
            cad_rads_category.get('Category for plaque burden', ''),
            cad_rads_category.get('Modifier N', ''),
            cad_rads_category.get('Modifier HRP', ''),
            cad_rads_category.get('Modifier S', ''),
            cad_rads_category.get('Modifier G', ''),
            cad_rads_category.get('Modifier E', '')
        ]

        modifiers_filtered = [mod for mod in modifiers if mod != "No" and mod != ""]
        modifiers_combined = "/".join(modifiers_filtered)

        results.append({
            'id': ids[i],
            'report': report_a,
            'category for stenosis severity': cad_rads_category.get('Category for stenosis severity', ''),
            'category for plaque burden': cad_rads_category.get('Category for plaque burden', ''),
            'Modifier N': cad_rads_category.get('Modifier N', ''),
            'Modifier HRP': cad_rads_category.get('Modifier HRP', ''),
            'Modifier S': cad_rads_category.get('Modifier S', ''),
            'Modifier G': cad_rads_category.get('Modifier G', ''),
            'Modifier E': cad_rads_category.get('Modifier E', ''),
            'Final category': modifiers_combined
        })

    output_df = pd.DataFrame(results)

    try:
        output_df.to_excel(output_file, index=False)
        print("Results have been successfully saved.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

input_file =  # input file
output_file =  # output file

if __name__ == "__main__":
    start_time = time.time()
    process_reports(input_file, output_file)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Elapsed time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
    
