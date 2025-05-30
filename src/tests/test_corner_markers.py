# test_corner_markers.py
import cv2
from corner_marker_aligner import CornerMarkerAligner

def test_alignment(
    original_path: str,
    scanned_path: str,
    out_dir: str      = "corner_results",
    test_id:  str     = "corner_demo",
    min_marker_area:  int = 400,
    debug: bool       = True
):
    # 1. cargar imágenes
    original = cv2.imread(original_path)
    scanned  = cv2.imread(scanned_path)
    if original is None or scanned is None:
        print("❌  No se pudo leer alguna de las imágenes.")
        return

    # 2. crear alineador
    aligner = CornerMarkerAligner(
        min_marker_area=min_marker_area,
        debug_mode=debug,
        debug_dir=out_dir
    )

    # 3. ejecutar
    aligned = aligner.align(original, scanned)

    # 4. guardar resultado
    if aligned is not None:
        cv2.imwrite(f"{out_dir}/aligned_{test_id}.png", aligned)
        print(f"✅  Alineación exitosa. Imágenes de debug en: {out_dir}/images/")
    else:
        print("❌  Falló la alineación. Revisa imágenes de debug (si existen).")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_alignment(
        original_path="src/tests/original_image/4.jpg",
        scanned_path ="src/tests/scanned_image/4_foar_721_2.jpg",
        out_dir      ="src/tests/corner_results",
        test_id      ="corner_fast",
        min_marker_area=400,
        debug=True
    )
