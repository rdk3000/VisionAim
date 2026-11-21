# VisionAim Desktop

Aplicativo Windows com interface própria e detecção visual YOLO11 Pose.

## Como obter o executável
1. Crie um repositório GitHub e envie o conteúdo desta pasta.
2. Abra a aba **Actions** do repositório.
3. Execute o workflow **Build VisionAim Windows**.
4. Baixe o artefato `VisionAim-Windows` ao final da execução.
5. Extraia e abra `VisionAim.exe`.

O build é feito em um runner Windows na nuvem; não exige instalar Python nem compilar no seu PC.
O pacote inclui o modelo YOLO11 Pose. A primeira execução pode demorar um pouco para iniciar.

## Escopo
A aplicação mostra captura de tela, caixas de detecção de pessoas e ponto do nariz.
Ela não move o mouse nem automatiza entradas.
