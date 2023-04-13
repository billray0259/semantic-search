from flask import Flask, render_template, request, jsonify, Markup
from document_embeddings import DocumentEmbeddings
import mistune
import pdf2image
from io import BytesIO
import base64
import config
import os

app = Flask(__name__)
app.config.from_object(config)

embeddings = DocumentEmbeddings.load('embeddings.npy', 'embeddings.json')



def render_txt(content):
    return Markup(f'<pre>{content}</pre>')

def render_md(content):
    html = mistune.markdown(content)
    return Markup(html)

def render_pdf(file_path, page_number):
    images = pdf2image.convert_from_path(file_path, first_page=page_number, last_page=page_number+1)
    img_bytes = BytesIO()
    images[0].save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    return Markup(f'<img src="data:image/png;base64,{img_b64}" alt="PDF page {page_number}" style="max-width: 100%;">')

render_functions = {
    '.txt': lambda file_path, page_number, content: render_txt(content),
    '.md': lambda file_path, page_number, content: render_md(content),
    '.pdf': lambda file_path, page_number, content: render_pdf(file_path, page_number)
}

def render_matched_content(file_path, page_number, content):
    _, ext = os.path.splitext(file_path)
    render_func = render_functions.get(ext)
    if render_func:
        return render_func(file_path, page_number, content)
    return Markup('<pre>Unsupported file type</pre>')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        n_results = request.form.get('n_results', config.NUM_RESULTS)
        sort_by = request.form.get('sort_by', 'relevance')
        file_types = request.form.getlist('file_types')

        results = embeddings.search(query, n_results=n_results)
        
        # Filter results by file types if provided
        if file_types:
            results = [result for result in results if os.path.splitext(result['source_file'])[1] in file_types]

        # Sort the results based on the provided sort option
        if sort_by == 'relevance':
            results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == 'date':
            results.sort(key=lambda x: x['file_last_modified'], reverse=True)

        # add the rendered content to the results
        results = [{
            **result,
            'rendered_content': render_matched_content(result['source_file'], result['page_number'], result['embedding_text']),
            'source_file': os.path.basename(result['source_file'])
        } for result in results]


        return render_template('index.html', results=results, query=query)

    return render_template('index.html')


def main():
    app.run(host='0.0.0.0', port=app.config['PORT'])

if __name__ == '__main__':
    main()