"""
Generator Imagini OBSID - Web Interface
=======================================
Flask app pentru generare imagini parfumuri
Deploy: Coolify @ imagini.obsid.ro
"""

import os
import sys
import json
import hashlib
import tempfile
import zipfile
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file, Response
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import requests

# rembg pentru eliminare fundal
try:
    from rembg import remove as rembg_remove
    REMBG_OK = True
except:
    REMBG_OK = False

app = Flask(__name__)

# ============== CONFIG ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CACHE_FOLDER = os.path.join(BASE_DIR, 'cache')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
FONTS_FOLDER = os.path.join(BASE_DIR, 'fonts')
TEMPLATES_FOLDER = os.path.join(BASE_DIR, 'templates_img')
CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')

# Creeaza foldere daca nu exista
for folder in [UPLOAD_FOLDER, CACHE_FOLDER, OUTPUT_FOLDER, FONTS_FOLDER, TEMPLATES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

DEFAULT_CONFIG = {
    "text_box": [352, 948, 118, 75],
    "image_box": [800, 619, 741, 880],
    "image_scale": 100
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_base_sku(sku):
    """Extrage SKU-ul de baza (fara -3, -5, -10)"""
    sku = str(sku)
    if sku.endswith('-10'):
        return sku[:-3]
    if sku.endswith('-3') or sku.endswith('-5'):
        return sku[:-2]
    return sku


def get_product_image(url):
    """Descarca si proceseaza imaginea (elimina fundal)"""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    nobg_file = os.path.join(CACHE_FOLDER, f"nobg_{url_hash}.png")
    cache_file = os.path.join(CACHE_FOLDER, f"{url_hash}.png")

    # Verifica cache nobg
    if os.path.exists(nobg_file):
        return Image.open(nobg_file).convert("RGBA")

    # Verifica cache original
    if os.path.exists(cache_file):
        img = Image.open(cache_file).convert("RGBA")
    else:
        try:
            resp = requests.get(url, timeout=30)
            img = Image.open(BytesIO(resp.content)).convert("RGBA")
            img.save(cache_file, "PNG")
        except Exception as e:
            print(f"Eroare descarcare imagine: {e}")
            return None

    # Elimina fundal cu rembg
    if REMBG_OK:
        try:
            img_bytes = BytesIO()
            img.save(img_bytes, 'PNG')
            img_bytes.seek(0)
            result = rembg_remove(img_bytes.read())
            img = Image.open(BytesIO(result)).convert("RGBA")
            img.save(nobg_file, "PNG")
        except Exception as e:
            print(f"Eroare rembg: {e}")

    return img


def create_image(template_path, font_path, product_name, product_url, config):
    """Creeaza imaginea finala"""
    template = Image.open(template_path).convert("RGBA")
    draw = ImageDraw.Draw(template)

    text_box = config['text_box']
    tx, ty, tw, th = text_box

    # Imparte pe randuri
    words = product_name.split()
    if len(words) >= 2:
        mid = len(words) // 2
        lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
    else:
        lines = [product_name]

    # Gaseste font size maxim care incape
    best_font = None
    for font_size in range(30, 7, -1):
        font = ImageFont.truetype(font_path, font_size)
        max_line_w = 0
        total_h = 0

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            max_line_w = max(max_line_w, bbox[2] - bbox[0])
            total_h += bbox[3] - bbox[1]

        total_h += 4 * (len(lines) - 1)

        if max_line_w <= tw and total_h <= th:
            best_font = font
            break

    if best_font is None:
        best_font = ImageFont.truetype(font_path, 8)

    # Calculeaza si deseneaza text centrat
    total_h = 0
    line_info = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=best_font)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        line_info.append((line, lw, lh))
        total_h += lh
    total_h += 4 * (len(lines) - 1)

    y_offset = ty + (th - total_h) // 2
    for line, lw, lh in line_info:
        x = tx + (tw - lw) // 2
        draw.text((x, y_offset), line, fill='black', font=best_font)
        y_offset += lh + 4

    # Imagine parfum
    if product_url:
        product_img = get_product_image(product_url)
        if product_img:
            ix, iy, iw, ih = config['image_box']
            scale_percent = config.get('image_scale', 100) / 100.0

            img_ratio = product_img.width / product_img.height
            box_ratio = iw / ih

            if img_ratio > box_ratio:
                new_w, new_h = iw, int(iw / img_ratio)
            else:
                new_h, new_w = ih, int(ih * img_ratio)

            # Aplica scalarea
            new_w = int(new_w * scale_percent)
            new_h = int(new_h * scale_percent)

            product_img = product_img.resize((new_w, new_h), Image.LANCZOS)
            paste_x = ix + (iw - new_w) // 2
            paste_y = iy + (ih - new_h) // 2
            template.paste(product_img, (paste_x, paste_y), product_img)

    return template.convert("RGB")


# ============== ROUTES ==============

@app.route('/')
def index():
    config = load_config()
    return render_template('index.html', config=config, rembg_ok=REMBG_OK)


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'GET':
        return jsonify(load_config())
    else:
        config = request.json
        save_config(config)
        return jsonify({"status": "ok"})


@app.route('/api/upload-template', methods=['POST'])
def upload_template():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename"}), 400

    filepath = os.path.join(TEMPLATES_FOLDER, 'template.jpg')
    file.save(filepath)
    return jsonify({"status": "ok", "path": filepath})


@app.route('/api/upload-font', methods=['POST'])
def upload_font():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename"}), 400

    filepath = os.path.join(FONTS_FOLDER, 'Montserrat-Bold.ttf')
    file.save(filepath)
    return jsonify({"status": "ok", "path": filepath})


@app.route('/api/upload-excel', methods=['POST'])
def upload_excel():
    excel_type = request.form.get('type', 'names')  # 'names' sau 'images'

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename"}), 400

    filename = 'excel_names.xlsx' if excel_type == 'names' else 'excel_images.xlsx'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Citeste si returneaza info
    try:
        df = pd.read_excel(filepath)
        return jsonify({
            "status": "ok",
            "path": filepath,
            "rows": len(df),
            "columns": df.columns.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/products')
def get_products():
    """Returneaza lista de produse din Excel"""
    excel_names = os.path.join(UPLOAD_FOLDER, 'excel_names.xlsx')
    excel_images = os.path.join(UPLOAD_FOLDER, 'excel_images.xlsx')

    if not os.path.exists(excel_names):
        return jsonify({"error": "Excel names not uploaded"}), 400

    try:
        df_names = pd.read_excel(excel_names)

        # Filtreaza doar -10
        df_filtered = df_names[df_names['Cod Produs (SKU)'].astype(str).str.endswith('-10')]

        # Construieste map URL-uri
        url_map = {}
        if os.path.exists(excel_images):
            df_images = pd.read_excel(excel_images)
            for _, row in df_images.iterrows():
                sku = str(row['Cod Produs (SKU)'])
                base = get_base_sku(sku)
                url = row.get('Imagine principala', None)
                if url and pd.notna(url) and base not in url_map:
                    url_map[base] = url

        products = []
        for _, row in df_filtered.iterrows():
            sku = str(row['Cod Produs (SKU)'])
            name = str(row['Denumire Produs']).strip()
            base = get_base_sku(sku)
            url = url_map.get(base, None)
            products.append({'sku': sku, 'name': name, 'url': url})

        return jsonify({"products": products, "total": len(products)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/preview', methods=['POST'])
def preview_image():
    """Genereaza preview pentru un produs"""
    data = request.json
    product_name = data.get('name', 'Test Produs')
    product_url = data.get('url', None)

    config = load_config()
    template_path = os.path.join(TEMPLATES_FOLDER, 'template.jpg')
    font_path = os.path.join(FONTS_FOLDER, 'Montserrat-Bold.ttf')

    if not os.path.exists(template_path):
        return jsonify({"error": "Template not uploaded"}), 400
    if not os.path.exists(font_path):
        return jsonify({"error": "Font not uploaded"}), 400

    try:
        img = create_image(template_path, font_path, product_name, product_url, config)

        # Returneaza ca base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)

        import base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({"image": f"data:image/jpeg;base64,{img_base64}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/generate-single', methods=['POST'])
def generate_single():
    """Genereaza o singura imagine si o returneaza pentru download"""
    data = request.json
    sku = data.get('sku', 'output')
    product_name = data.get('name', 'Test')
    product_url = data.get('url', None)

    config = load_config()
    template_path = os.path.join(TEMPLATES_FOLDER, 'template.jpg')
    font_path = os.path.join(FONTS_FOLDER, 'Montserrat-Bold.ttf')

    try:
        img = create_image(template_path, font_path, product_name, product_url, config)

        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)

        return send_file(buffer, mimetype='image/jpeg',
                        as_attachment=True,
                        download_name=f'{sku}.jpg')

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/export-all', methods=['POST'])
def export_all():
    """Exporta toate imaginile intr-un ZIP"""
    config = load_config()
    template_path = os.path.join(TEMPLATES_FOLDER, 'template.jpg')
    font_path = os.path.join(FONTS_FOLDER, 'Montserrat-Bold.ttf')

    if not os.path.exists(template_path):
        return jsonify({"error": "Template not uploaded"}), 400
    if not os.path.exists(font_path):
        return jsonify({"error": "Font not uploaded"}), 400

    # Preia lista de produse
    excel_names = os.path.join(UPLOAD_FOLDER, 'excel_names.xlsx')
    excel_images = os.path.join(UPLOAD_FOLDER, 'excel_images.xlsx')

    if not os.path.exists(excel_names):
        return jsonify({"error": "Excel names not uploaded"}), 400

    try:
        df_names = pd.read_excel(excel_names)
        df_filtered = df_names[df_names['Cod Produs (SKU)'].astype(str).str.endswith('-10')]

        url_map = {}
        if os.path.exists(excel_images):
            df_images = pd.read_excel(excel_images)
            for _, row in df_images.iterrows():
                sku = str(row['Cod Produs (SKU)'])
                base = get_base_sku(sku)
                url = row.get('Imagine principala', None)
                if url and pd.notna(url) and base not in url_map:
                    url_map[base] = url

        # Creeaza ZIP in memorie
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for _, row in df_filtered.iterrows():
                sku = str(row['Cod Produs (SKU)'])
                name = str(row['Denumire Produs']).strip()
                base = get_base_sku(sku)
                url = url_map.get(base, None)

                try:
                    img = create_image(template_path, font_path, name, url, config)

                    img_buffer = BytesIO()
                    img.save(img_buffer, format='JPEG', quality=95)
                    img_buffer.seek(0)

                    zf.writestr(f'{sku}.jpg', img_buffer.getvalue())
                except Exception as e:
                    print(f"Eroare la {sku}: {e}")

        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip',
                        as_attachment=True,
                        download_name='imagini_export.zip')

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/generate-custom', methods=['POST'])
def generate_custom():
    """Genereaza imagine cu nume si URL custom"""
    data = request.json
    sku = data.get('sku', 'custom')
    product_name = data.get('name', '')
    product_url = data.get('url', '')

    if not product_name:
        return jsonify({"error": "Numele produsului este obligatoriu"}), 400

    config = load_config()
    template_path = os.path.join(TEMPLATES_FOLDER, 'template.jpg')
    font_path = os.path.join(FONTS_FOLDER, 'Montserrat-Bold.ttf')

    if not os.path.exists(template_path):
        return jsonify({"error": "Template not uploaded"}), 400
    if not os.path.exists(font_path):
        return jsonify({"error": "Font not uploaded"}), 400

    try:
        img = create_image(template_path, font_path, product_name, product_url if product_url else None, config)

        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)

        return send_file(buffer, mimetype='image/jpeg',
                        as_attachment=True,
                        download_name=f'{sku}.jpg')

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/template-image')
def get_template_image():
    """Returneaza template-ul pentru editor zone"""
    template_path = os.path.join(TEMPLATES_FOLDER, 'template.jpg')
    if os.path.exists(template_path):
        return send_file(template_path, mimetype='image/jpeg')
    return jsonify({"error": "Template not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
