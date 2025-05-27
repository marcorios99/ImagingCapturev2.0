USE ENLA2024
GO

CREATE TABLE [dbo].[Plantilla](
    PlantillaID int IDENTITY(1,1) NOT NULL,
    Nombre varchar(255) NOT NULL,
    RutaImagen varchar(500) NULL,
    NroPaginas smallint NULL,
    FechaCreacion datetime2 NOT NULL DEFAULT GETDATE(),
    FechaModificacion datetime2 NULL,
    CONSTRAINT PK_Plantilla PRIMARY KEY CLUSTERED ([PlantillaID])
)
GO

CREATE TABLE [dbo].[TipoCampo](
    TipoCampoID int IDENTITY(1,1) NOT NULL,
    Nombre varchar(15) UNIQUE NOT NULL,
    Descripcion varchar(128) NULL,
    CONSTRAINT PK_TipoCampo PRIMARY KEY CLUSTERED (TipoCampoID)
)
GO

INSERT INTO [dbo].[TipoCampo] (Nombre, Descripcion) VALUES 
('OMR', 'Reconocimiento de marcas ópticas'),
('ICR', 'Reconocimiento inteligente de caracteres'),
('BARCODE', 'Código de barras'),
('XMARK', 'Marcas X')
GO

CREATE TABLE [dbo].[Campo](
    CampoID int IDENTITY(1,1) NOT NULL,
    PlantillaID int NOT NULL,
    Pregunta varchar(100) NOT NULL,
    TipoCampoID int NULL,
    PaginaNumero smallint NOT NULL,
    X float NOT NULL,
    Y float NOT NULL,
    Ancho float NOT NULL,
    Alto float NOT NULL,
    UmbralLocal tinyint NULL, 
    Validaciones varchar(255) NULL, -- REGEX con reglas de validación
    Indice [smallint] NULL,
    CONSTRAINT PK_Campo PRIMARY KEY CLUSTERED (CampoID),
	CONSTRAINT FK_Campo_TipoCampo FOREIGN KEY (TipoCampoID) 
    REFERENCES TipoCampo (TipoCampoID),
    CONSTRAINT FK_Campo_Plantilla FOREIGN KEY (PlantillaID) 
    REFERENCES Plantilla (PlantillaID)
) 
GO

CREATE TABLE [dbo].[Examen](
    ExamenID int IDENTITY(1,1) NOT NULL,
    CodigoBarras varchar(50) NOT NULL,
    PlantillaID int NOT NULL,
    FechaCreacion datetime NOT NULL DEFAULT GETDATE(),
    FechaModificacion datetime2 NULL,
    Estado char(1) NOT NULL CHECK (Estado IN ('0', '1', '2', '3')) -- 0:Pendiente, 1:Procesado, 2:Revisado (CC), 3:Finalizado
    CONSTRAINT PK_Examen PRIMARY KEY CLUSTERED (ExamenID),
    CONSTRAINT UK_Examen_CodigoBarras UNIQUE (CodigoBarras),
    CONSTRAINT FK_Examen_Plantilla FOREIGN KEY (PlantillaID) 
    REFERENCES Plantilla (PlantillaID),)
GO

CREATE TABLE [dbo].[Imagen](
    ImagenID int IDENTITY(1,1) NOT NULL,
    ExamenID int NOT NULL,
    RutaImagen varchar(500) NOT NULL,
    NumeroPagina int NOT NULL DEFAULT 1,
    FechaCreacion [datetime] NOT NULL DEFAULT GETDATE(),
	SumaHash varchar(40),
	Alineado bit DEFAULT 0,
	Estado char(1) NOT NULL CHECK (Estado IN ('0', '1', '2', '3')) -- 0:Pendiente, 1:Procesado, 2:Revisado (CC), 3:Finalizado
    CONSTRAINT PK_ExamenImagen PRIMARY KEY CLUSTERED (ImagenID),
    CONSTRAINT FK_ExamenImagen_Examen FOREIGN KEY (ExamenID) 
    REFERENCES Examen (ExamenID)
)
GO

CREATE TABLE [dbo].[Resultados](
    ResultadoID bigint IDENTITY(1,1) NOT NULL,
    ImagenID int NOT NULL,
    CampoID int NOT NULL,
    Certeza tinyint NULL,
    FechaProceso datetime NOT NULL,
    CONSTRAINT PK_Resultados PRIMARY KEY NONCLUSTERED (ResultadoID),
	CONSTRAINT FK_Resultados_Imagen FOREIGN KEY (ImagenID) 
	REFERENCES Imagen (ImagenID),
	CONSTRAINT FK_Resultados_Campo FOREIGN KEY (CampoID) 
    REFERENCES Campo (CampoID)
) 
GO