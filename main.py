import cv2
import numpy as np
import utlis


#
webCamFeed = True
pathImage = "5.jpg"
cap = cv2.VideoCapture(1)
cap.set(10,160)
heightImg = 700
widthImg  = 700
questions=5
choices=5
ans= [1,2,0,2,4]
#


count=0

while True:

    if webCamFeed:success, img = cap.read()
    else:img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  # Redimensionar a imagem
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Criar uma imagem em branco para depuração
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converter imagem para escala de cinza
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Aplicar desfoque gaussiano
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # Aplicar detecção de bordas com Canny

    try:
        ## Encontrar todos os contornos
        imgContours = img.copy()  # Copiar imagem para exibir contornos
        imgBigContour = img.copy()  # Copiar imagem para exibir maiores contornos
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Encontrar contornos
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # Desenhar todos os contornos detectados
        rectCon = utlis.rectContour(contours)  # Filtrar contornos retangulares
        biggestPoints = utlis.getCornerPoints(rectCon[0])  # Obter pontos de canto do maior retângulo
        gradePoints = utlis.getCornerPoints(rectCon[1])  # Obter pontos de canto do segundo maior retângulo

        

        if biggestPoints.size != 0 and gradePoints.size != 0:

            # Ajustar o maior retângulo
            biggestPoints = utlis.reorder(biggestPoints)  # Reordenar pontos para a perspectiva
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)  # Desenhar maior contorno
            pts1 = np.float32(biggestPoints)  # Preparar pontos para transformação de perspectiva
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # Pontos de destino
            matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Obter matriz de transformação
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # Aplicar transformação

            # Ajustar o segundo maior retângulo
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)  # Desenhar contorno do segundo maior retângulo
            gradePoints = utlis.reorder(gradePoints)  # Reordenar pontos
            ptsG1 = np.float32(gradePoints)  # Preparar pontos para transformação
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # Pontos de destino
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)  # Obter matriz de transformação
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))  # Aplicar transformação

            # Aplicar Threshold
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]  # Aplicar Threshold invertida

            boxes = utlis.splitBoxes(imgThresh) # Dividir em caixas individuais
            cv2.imshow("Split Test ", boxes[3])
            countR=0
            countC=0
            myPixelVal = np.zeros((questions,choices)) # Armazenar valores de pixel de cada caixa
            for image in boxes:
                #cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC]= totalPixels
                countC += 1
                if (countC==choices):countC=0;countR +=1

            # Encontrar respostas do usuário e compará-las com as corretas
            myIndex=[]
            for x in range (0,questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            #print("USER ANSWERS",myIndex)

            grading=[]
            for x in range(0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)   # Resposta correta
                else:grading.append(0)  # Resposta incorreta
            #print("GRADING",grading)
            score = (sum(grading)/questions)*100 # Calcular pontuação final
            #print("SCORE",score)

            # Exibir respostas e pontuação
            utlis.showAnswers(imgWarpColored,myIndex,grading,ans) # Mostrar respostas detectadas
            utlis.drawGrid(imgWarpColored) # Desenhar grade
            imgRawDrawings = np.zeros_like(imgWarpColored) # Nova imagem em branco
            utlis.showAnswers(imgRawDrawings, myIndex, grading, ans) # Desenhar respostas na nova imagem
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # Inverter transformação
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) # Aplicar inversão

            # Display
            imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # Nova imagem para exibir nota
            cv2.putText(imgRawGrade,str(int(score))+"%",(70,100)
                        ,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3) # Adicionar pontuação
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) # Inverter transformação
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) # Aplicar inversão

            
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0) # Combinar imagem final
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0) # Combinar nota

            imageArray = ([img,imgGray,imgCanny,imgContours],
                          [imgBigContour,imgThresh,imgWarpColored,imgFinal])
            cv2.imshow("Final Result", imgFinal) # Mostrar imagem final
    except:
        imageArray = ([img,imgGray,imgCanny,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    lables = [["Original","Gray","Edges","Contours"],
              ["Biggest Contour","Threshold","Warpped","Final"]]

    
    # Exibir as imagens lado a lado
    stackedImage = utlis.stackImages(imageArray,0.5,lables)
    cv2.imshow("Result",stackedImage)

    # Salvar a imagem quando a tecla 's' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1