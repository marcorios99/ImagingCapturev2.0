WITH Coordenadas AS (
    SELECT 
        2843 AS X1,
        2185 AS Y1,
        3445 AS X2,
        2350 AS Y2,
        3445 - 2843 AS Ancho,
        2350 - 2185 AS Alto
)
INSERT INTO Campo (PlantillaID, Pregunta, TipoCampoID, PaginaNumero, X, Y, Ancho, Alto, Validaciones)
SELECT 
    p.PlantillaID, 
    'BARCODE' AS Pregunta, 
    NULL AS TipoCampoID, 
    1 AS PaginaNumero, 
    c.X1 AS X, 
    c.Y1 AS Y, 
    c.Ancho, 
    c.Alto, 
    null AS Validaciones
FROM 
    Plantilla p, 
    Coordenadas c;