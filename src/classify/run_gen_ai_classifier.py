import pandas as pd, os, zipfile, json, tiktoken, statistics
from openai import OpenAI
from pydantic import BaseModel
from datetime import datetime

# Output class component
class ReturnComponent(BaseModel):
    index: int
    brand: str
    code: float
    confidence: float
    note: str

# Output class - this is used to enforce the structure of the output
class ClassificationReturn(BaseModel):
    values: list[ReturnComponent]

def dataframe_to_api_records(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame into the JSON string for the `records` argument
    of the classify_brands function.
    """
    records = []
    # Iterate over rows in the DataFrame
    for idx, row in df.iterrows():
        # Collect up to three description fields (adjust if you have more/less)
        descriptions = [
            row.get('description1'),
            row.get('description2'),
            row.get('description3')
        ]
        # Filter out any missing or NaN descriptions
        descriptions = [desc for desc in descriptions if pd.notna(desc)]
        
        record = {
            "index": int(idx),
            "brand": row['brand'],
            "descriptions": descriptions
        }
        records.append(record)
    
    # Convert to a JSON-formatted string
    return json.dumps({"records": records}, ensure_ascii=False)

def get_chatgpt_return(client, model, system_message_content, in_df):
    records_json = dataframe_to_api_records(in_df)

    response = client.responses.parse(
        model=model,
        input=[
            {"role":"system", "content": system_message_content},
            {
                "role": "user",
                "content": records_json,
            },
        ],
        text_format=ClassificationReturn,
    )
    event = response.output_parsed
    out_df = pd.DataFrame([comp.model_dump() for comp in event.values]).set_index('index', drop=True)   
    return response, out_df


try: # Check we have the API key installed
    apiKey = os.getenv("OPENAI_API_KEY")
    print(f'Found API key with length {len(apiKey)}')
except:
    print('Cannot find OPENAI_API_KEY environment variable')

client = OpenAI()
model = "gpt-4o"  # More expensive, older and doesn't support structured outputs or batching
model = "o4-mini" # Use this - it is slightly slower, but has more recent training data and supports structured outputs
system_message_content =  open("system_message.txt", "r", encoding="utf-8").read()
records_per_call = 80

# Get the top 20K brands, by total debit spend
df = pd.read_parquet('top_20K.parquet')
print('Have top 20K brands. Pushing through OpenAI API')

# Push all of those through the API
rdf = []
responses = []
run_start = datetime.now()
rs_txt = run_start.strftime('%Y-%m-%d_%H-%M-%S')
for i in range(0, len(df), records_per_call):
    in_df = df.iloc[i:i + records_per_call]

    response, out_df = get_chatgpt_return(client, model, system_message_content, in_df)

    rdf += [pd.merge(out_df, in_df, left_index=True, right_index=True)]
    responses += [response]
    
    print(f'''ChatGPT returned for the {(i+records_per_call)/records_per_call
                                      }th time, after {
                                          (datetime.now() - run_start).seconds
                                                       } seconds and {(i+records_per_call)} records''')
    tmp_out = pd.concat(rdf)
    misalignments = sum(tmp_out.brand_x != tmp_out.brand_y)
    if misalignments!=0:
        print(80*'#',f'\nWARNING: {misalignments} misaligned brands\n', 80*'#')

    # Output
    tmp_out.to_excel(f'OpenAI_classifications_{rs_txt}.xlsx')
    with open(f'responses_{rs_txt}.json', 'w') as f:
        json.dump([r.model_dump() for r in responses], f)