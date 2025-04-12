import openai
import csv

def generate_chat_responses(prompt, n, api_key, output_file='gen_text.csv'):
    openai.api_key = api_key

    responses = []
    for i in range(n):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        responses.append((i, response.choices[0].message['content']))

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Text'])
        writer.writerows(responses)

# 使用示例
# 这里修改提示语，下面例子是我们数据集第0个提示语
prompt = "Write a persuasive essay about the benefits of recycling in schools. Your essay must be based on ideas and information that can be found in the passage set. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write your essay in the space provided."
n = 50  # 生成响应的数量（你想根据这个提示语生成多少个样本）
api_key = "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"  # 替换为你自己的的openai的 API 密钥
generate_chat_responses(prompt, n, api_key)