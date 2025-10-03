import streamlit as st
import cv2
import numpy as np

# --- FUNÇÕES DE PROCESSAMENTO EXISTENTES (Com ajuste do Threshold para 200) ---

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Limiar ajustado para 200 para capturar marcas mais claras
    thresh = cv2.threshold(blur, 200, 225, cv2.THRESH_BINARY_INV)[1]
    return thresh

def detect_bubbles(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in contours:
        area = cv2.contourArea(c)
        if 300 < area < 3000:
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h)
            if 0.8 <= ratio <= 1.2:
                bubbles.append((x, y, w, h, c))
    return bubbles

def group_bubbles_by_columns(bubbles, num_columns=4):
    if not bubbles:
        return []

    bubbles = sorted(bubbles, key=lambda b: b[0])
    min_x, max_x = min([b[0] for b in bubbles]), max([b[0] for b in bubbles])
    col_width = (max_x - min_x) / num_columns

    columns = [[] for _ in range(num_columns)]
    for b in bubbles:
        x = b[0]
        col_index = min(int((x - min_x) / (col_width + 5)), num_columns - 1)
        columns[col_index].append(b)

    all_questions = []
    for col in columns:
        col_sorted = sorted(col, key=lambda b: b[1])
        line = []
        last_y = None
        tol = 25
        for b in col_sorted:
            x, y, w, h, c = b
            if last_y is None or abs(y - last_y) < tol:
                line.append(b)
                last_y = y
            else:
                all_questions.append(sorted(line, key=lambda bb: bb[0]))
                line = [b]
                last_y = y
        if line:
            all_questions.append(sorted(line, key=lambda bb: bb[0]))

    return all_questions

def get_marked_alternatives(questions, thresh):
    answers = []
    for q in questions:
        if len(q) != 5:
            answers.append(None)
            continue
        intensities = []
        for (x, y, w, h, c) in q:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean = cv2.mean(thresh, mask=mask)[0]
            intensities.append(mean)
        marked = np.argmax(intensities)
        answers.append(marked)
    return answers

def avaliar_e_desenhar(imagem, questions, answers, gabarito):
    options = ["A", "B", "C", "D", "E"]
    acertos = 0

    # Cria uma cópia para evitar modificar a imagem original
    imagem_resultado = imagem.copy() 

    for i, ans in enumerate(answers):
        num_questao = i + 1
        
        if ans is None or num_questao not in gabarito:
            continue

        marcado = options[ans]
        correto = gabarito[num_questao]
        cor = (0, 0, 255)  # vermelho = errado
        if marcado == correto:
            cor = (0, 255, 0)  # verde = certo
            acertos += 1

        x, y, w, h, c = questions[i][ans]
        cv2.rectangle(imagem_resultado, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(imagem_resultado, f"{num_questao}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

    # Adiciona a pontuação no canto superior
    texto_acertos = f"Acertos: {acertos}/{len(gabarito)}"
    
    cv2.putText(imagem_resultado, texto_acertos, (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(imagem_resultado, texto_acertos, (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return imagem_resultado, acertos

# --- LÓGICA PRINCIPAL DO STREAMLIT ---

def app():
    st.set_page_config(page_title="Corretor OMR Web", layout="centered")
    
    # Gabarito (Hardcoded)
    gabarito = {
        1: "C", 2: "D", 3: "D", 4: "C", 5: "E", 6: "C", 7: "B", 8: "C", 9: "B", 10: "E",
        11: "C", 12: "D", 13: "B", 14: "A", 15: "A", 16: "D", 17: "E", 18: "C", 19: "D", 20: "C",
        21: "B", 22: "A", 23: "C", 24: "E", 25: "D", 26: "D", 27: "B", 28: "B", 29: "E", 30: "D",
        31: "A", 32: "D", 33: "D", 34: "C", 35: "D", 36: "A", 37: "A", 38: "E", 39: "B", 40: "E",
        41: "B", 42: "E", 43: "C", 44: "C", 45: "C", 46: "C", 47: "E", 48: "C", 49: "B", 50: "E",
        51: "E", 52: "A", 53: "A", 54: "E", 55: "D", 56: "E", 57: "C", 58: "C", 59: "A", 60: "B",
        61: "E", 62: "B", 63: "D", 64: "E", 65: "E", 66: "A", 67: "D", 68: "B", 69: "C", 70: "C"
    }

    st.title("Gabarito Web - Corretor OMR")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("1. Faça o upload da imagem da folha de respostas digitalizada", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        
        # Carrega e decodifica o arquivo enviado pelo Streamlit
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # 2. Redimensionamento da folha inteira para tamanho consistente (1000px)
        MAX_PROCESSING_WIDTH = 1000 
        
        if frame.shape[1] > MAX_PROCESSING_WIDTH:
            scale_factor = MAX_PROCESSING_WIDTH / frame.shape[1]
            new_height = int(frame.shape[0] * scale_factor)
            roi_processed = cv2.resize(frame, (MAX_PROCESSING_WIDTH, new_height))
        else:
            roi_processed = frame.copy() 

        # 3. Processamento, detecção e correção
        with st.spinner('Corrigindo a folha...'):
            thresh = preprocess_image(roi_processed)
            bubbles = detect_bubbles(thresh)
            questions = group_bubbles_by_columns(bubbles)
            
            if not questions:
                st.error("Erro: Nenhuma folha de respostas ou bolha detectada na imagem. Certifique-se de que a imagem está nítida e bem contrastada.")
                st.image(cv2.cvtColor(roi_processed, cv2.COLOR_BGR2RGB), caption="Imagem processada, mas sem detecção de bolhas.")
            else:
                answers = get_marked_alternatives(questions, thresh)
                
                # Obtém a imagem corrigida e a pontuação
                resultado, acertos = avaliar_e_desenhar(roi_processed.copy(), questions, answers, gabarito)
                
                # Exibe o resultado na interface web
                st.markdown("## 2. Resultado da Correção")
                st.success(f"### 🎉 Pontuação Final: {acertos}/{len(gabarito)} Acertos")

                # Exibe a imagem (OpenCV usa BGR, Streamlit espera RGB)
                st.image(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB), caption="Folha Corrigida com Marcações", use_container_width=True)

                # Funcionalidade para salvar (Download)
                is_success, buffer = cv2.imencode(".jpg", resultado)
                if is_success:
                    st.download_button(
                        label="Clique para salvar a imagem corrigida (.jpg)",
                        data=buffer.tobytes(),
                        file_name="gabarito_corrigido.jpg",
                        mime="image/jpeg"
                    )

if __name__ == "__main__":
    app()