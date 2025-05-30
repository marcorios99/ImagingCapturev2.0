import cv2
from align_omr_test import FastOMRAligner     # ← donde hayas guardado la clase

def test_alignment(
    original_path: str,
    scanned_path: str,
    out_dir: str      = "resultados",
    debug_id: str     = "prueba_001",
    min_area: int     = 900000
):
    # 1. cargar imágenes
    original = cv2.imread(original_path)
    scanned  = cv2.imread(scanned_path)

    if original is None or scanned is None:
        print("❌  No se pudo leer alguna de las imágenes.")
        return

    # 2. crear alineador
    aligner = FastOMRAligner(
        debug_mode=True,
        debug_dir=out_dir,
        min_area=min_area
    )

    # 3. ejecutar
    aligned = aligner.align(
        original=original,
        scanned=scanned,
        debug_id=debug_id
    )

    if aligned is not None:
        cv2.imwrite(f"{out_dir}/aligned_{debug_id}.png", aligned)
        print("✅  Alineación exitosa. Resultado guardado.")
    else:
        print("❌  Falló la alineación. Revisa la carpeta de debug.")

# --------------------------------------------------------------------
if __name__ == "__main__":
    test_alignment(
        original_path="src/tests/original_image/4.jpg",
        scanned_path ="src/tests/scanned_image/4_foar_721_2.jpg",
        out_dir      ="src/tests/resultados",
        debug_id     ="src/tests/demo_fast",
        min_area     =900000
    )
