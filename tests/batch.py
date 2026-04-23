import pandas as pd

batch_info = [
    ['51_PRJNA352399', 4], ['34_PRJNA543751', 6], ['39_PRJNA545420', 8],
    ['48_PRJNA379540', 4], ['54_PRJNA272532', 2], ['22_PRJNA684913', 6],
    ['9_PRJNA830513', 12], ['29_PRJNA644619', 4], ['21_PRJNA770634', 6],
    ['3_PRJNA902757', 18], ['5_PRJNA884833', 14], ['20_PRJNA768278', 6],
    ['47_PRJNA433173', 8], ['12_PRJNA728071', 6], ['2_PRJNA785886', 4],
    ['19_PRJNA509432', 12], ['11_PRJNA798412', 4], ['36_PRJNA413707', 6],
    ['10_PRJNA787205', 18], ['32_PRJNA613097', 11], ['15_PRJNA801504', 2],
    ['37_PRJNA545412', 5], ['4_PRJNA760304', 6], ['28_PRJNA703019', 11],
    ['18_PRJNA732706', 12], ['40_PRJNA516578', 6], ['45_PRJNA392253', 27],
    ['6_PRJNA760308', 6], ['46_PRJNA398489', 2], ['8_PRJNA715173', 18],
    ['31_PRJNA637714', 9], ['14_PRJNA801620', 2], ['35_PRJNA623616', 12],
    ['26_PRJNA726423', 2], ['7_PRJNA821209', 6], ['38_PRJNA492802', 6],
    ['13_PRJNA828148', 8], ['1_PRJNA878684', 6], ['49_PRJNA392174', 5],
    ['33_PRJNA645064', 8], ['53_PRJNA328520', 12], ['52_PRJNA344668', 4],
    ['17_PRJNA793888', 6], ['42_PRJNA514926', 6], ['44_PRJNA436809', 8],
    ['16_PRJNA800015', 6], ['30_PRJNA686868', 2], ['50_PRJNA355247', 6],
    ['43_PRJNA464389', 36], ['24_PRJNA689806', 6], ['27_PRJNA725424', 12],
    ['23_PRJNA726422', 4]
]

def clean_batch_name(name: str) -> str:
    parts = name.split("_", 1)
    return parts[1] if len(parts) > 1 else name

rows = []
sample_id = 1

for batch_name, count in batch_info:
    clean_batch = clean_batch_name(batch_name)

    for _ in range(count):
        rows.append({
            "sample": f"sample_{sample_id}",
            "batch": clean_batch
        })
        sample_id += 1

batch_df = pd.DataFrame(rows)
batch_df.to_csv("ab_batch_expanded_clean.csv", index=False)

print(batch_df.head(20))
print("总样本数：", len(batch_df))