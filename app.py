import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. ARCHITECTURE DU MODÈLE (Attention U-Net)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, g, x):
        psi = F.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.e1 = ConvBlock(3, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.b = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att4 = AttentionGate(512, 512, 256)
        self.d4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = AttentionGate(256, 256, 128)
        self.d3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionGate(128, 128, 64)
        self.d2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = AttentionGate(64, 64, 32)
        self.d1 = ConvBlock(128, 64)

        self.out = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.e1(x); p1 = self.pool(x1)
        x2 = self.e2(p1); p2 = self.pool(x2)
        x3 = self.e3(p2); p3 = self.pool(x3)
        x4 = self.e4(p3); p4 = self.pool(x4)
        b = self.b(p4)

        d4 = self.up4(b); x4 = self.att4(d4, x4); d4 = torch.cat((x4, d4), dim=1); d4 = self.d4(d4)
        d3 = self.up3(d4); x3 = self.att3(d3, x3); d3 = torch.cat((x3, d3), dim=1); d3 = self.d3(d3)
        d2 = self.up2(d3); x2 = self.att2(d2, x2); d2 = torch.cat((x2, d2), dim=1); d2 = self.d2(d2)
        d1 = self.up1(d2); x1 = self.att1(d1, x1); d1 = torch.cat((x1, d1), dim=1); d1 = self.d1(d1)

        return self.out(d1)

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNet().to(device)


try:
    model.load_state_dict(torch.load("models/attention_unet_fissures.pth", map_location=device))
    print("Poids du modèle chargés avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle. Vérifiez le chemin : {e}")

model.eval()

# ==========================================
# 3. RÈGLES MÉTIER ET POST-TRAITEMENT
# ==========================================
def evaluer_fissure(prediction_masque_brut):
    pred_img = (prediction_masque_brut * 255).astype(np.uint8)
    _, masque_binaire = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    masque_propre = cv2.morphologyEx(masque_binaire, cv2.MORPH_CLOSE, kernel)

    surface_totale = masque_propre.shape[0] * masque_propre.shape[1]
    pixels_fissure = cv2.countNonZero(masque_propre)
    densite = (pixels_fissure / surface_totale) * 100

    contours, _ = cv2.findContours(masque_propre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    longueur_max, largeur_max = 0, 0
    if contours:
        for cnt in contours:
            longueur = cv2.arcLength(cnt, False) / 2
            if longueur > longueur_max:
                longueur_max = longueur
                aire = cv2.contourArea(cnt)
                largeur_max = aire / longueur if longueur > 0 else 0

    severite = "Saine"
    if densite > 5 or largeur_max > 10:
        severite = "Critique (Intervention urgente)"
    elif densite > 1 or longueur_max > 50:
        severite = "Moyenne (A surveiller)"
    elif pixels_fissure > 0:
        severite = "Légère"

    return {
        "Densité (%)": round(densite, 2),
        "Longueur max (px)": round(longueur_max, 2),
        "Largeur max (px)": round(largeur_max, 2),
        "Sévérité": severite
    }

# ==========================================
# 4. FONCTION D'INFÉRENCE GRADIO
# ==========================================
def inspecter_route(image_numpy):
    img_resized = cv2.resize(image_numpy, (256, 256))
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std

    tensor_img = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        prediction = model(tensor_img)
        masque_brut = prediction.squeeze().cpu().numpy()

    rapport = evaluer_fissure(masque_brut)

    masque_binaire = (masque_brut > 0.5).astype(np.uint8) * 255
    calque_rouge = np.zeros_like(img_resized)
    calque_rouge[:, :, 0] = masque_binaire 

    image_resultat = cv2.addWeighted(img_resized, 0.8, calque_rouge, 0.8, 0)

    texte_final = f"🚨 SÉVÉRITÉ : {rapport['Sévérité']}\n"
    texte_final += f"━" * 30 + "\n"
    texte_final += f"📏 Longueur max : {rapport['Longueur max (px)']} pixels\n"
    texte_final += f"↔️ Largeur max : {rapport['Largeur max (px)']} pixels\n"
    texte_final += f"🕸️ Densité de dégradation : {rapport['Densité (%)']} %"

    return image_resultat, texte_final

# ==========================================
# 5. LANCEMENT DE L'INTERFACE
# ==========================================
interface = gr.Interface(
    fn=inspecter_route,
    inputs=gr.Image(label="1. Uploadez une image de route (Cameroun ou CFD)"),
    outputs=[
        gr.Image(label="2. Détection Attention U-Net"),
        gr.Textbox(label="3. Rapport d'Inspection Technique", lines=6)
    ],
    title="🚧 Système d'Inspection des Infrastructures par IA",
    description="Modèle Attention U-Net. Détection au pixel près et règles métiers de priorisation.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch(debug=False)
