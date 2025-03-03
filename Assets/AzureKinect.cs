using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.Azure.Kinect.Sensor;
using System.Threading.Tasks;
using System.Linq;
using System.Runtime.InteropServices;

public class MultiKinectMeshScript : MonoBehaviour
{
    [Header("レンダリング設定")]
    public Material pointCloudMaterial;
    public float pointSize = 5.0f;
    public MeshTopology meshTopology = MeshTopology.Points;

    [Header("メッシュ品質の設定")]
    [Tooltip("深度の不連続性を検出するしきい値。大きくすると接続が増える")]
    public float depthDiscontinuityThreshold = 0.05f;
    [Tooltip("ノイズ除去のフィルタを有効にする")]
    public bool enableNoiseFiltering = true;
    [Tooltip("中央値からの外れ値を検出するしきい値。大きくするとノイズ除去が弱くなる")]
    public float outlierThreshold = 0.1f;

    [Header("GPU高速化")]
    [Tooltip("GPUを使用して処理を高速化")]
    public bool useGPUAcceleration = true;
    public ComputeShader pointCloudProcessor;

    [Header("キャリブレーション設定")]
    public bool useTransposedRotation = true;
    public Vector3 kinect2ManualPosition = new Vector3(-0.39f, 0.11f, -1.13f);
    public Vector3 kinect2ManualRotation = new Vector3(0.167f, -0.801f, -4.548f);

    // Kinectデバイス
    private Device kinect1;
    private Device kinect2;
    private Transformation transformation1;
    private Transformation transformation2;
    private Matrix4x4 transform2To1;

    // Kinect1用Meshとデータ
    private Mesh mesh1;
    private Vector3[] vertices1;
    private Color32[] colors1;
    private int num1, width1, height1;

    // Kinect2用Meshとデータ
    private Mesh mesh2;
    private Vector3[] vertices2;
    private Color32[] colors2;
    private int num2, width2, height2;
    private GameObject kinect2Object;

    // GPUリソース
    private ComputeBuffer inputVertexBuffer1, outputVertexBuffer1;
    private ComputeBuffer validityMaskBuffer1, indirectArgsBuffer1, counterBuffer1;
    private ComputeBuffer inputVertexBuffer2, outputVertexBuffer2;
    private ComputeBuffer validityMaskBuffer2, indirectArgsBuffer2, counterBuffer2;
    private int filterDepthMapKernelId, generateVertexBufferKernelId;

    // データバッファ
    private Short3[] xyzBuffer1, xyzBuffer2;
    private BGRA[] colorBuffer1, colorBuffer2;

    // パフォーマンス測定
    private float frameTime, fps;
    private int frameCount;

    void Start()
    {
        InitializeMaterial();
        InitKinectDevices();
        InitMesh1();
        InitMesh2();
        SetupExternalTransformation();

        if (useGPUAcceleration && pointCloudProcessor != null)
        {
            InitGPUResources();
        }
        else if (useGPUAcceleration)
        {
            Debug.LogWarning("ComputeShaderが設定されていないため、GPU高速化を無効化します。");
            useGPUAcceleration = false;
        }

        // 非同期点群処理開始
        Task t1 = KinectLoop1();
        Task t2 = KinectLoop2();
    }

    private void InitializeMaterial()
    {
        if (pointCloudMaterial == null)
        {
            Shader shader = Shader.Find("Custom/ColoredVertex");
            if (shader != null)
            {
                pointCloudMaterial = new Material(shader);
                pointCloudMaterial.SetFloat("_PointSize", pointSize);
                pointCloudMaterial.SetFloat("_Size", 1.0f);
            }
            else
            {
                Debug.LogError("Custom/ColoredVertex シェーダーが見つかりません。");
                shader = Shader.Find("Standard");
                if (shader != null)
                {
                    Debug.LogWarning("Standard シェーダーを使用します。");
                    pointCloudMaterial = new Material(shader);
                }
                else
                {
                    Debug.LogError("シェーダーが見つかりません。マテリアルを手動で設定してください。");
                    return;
                }
            }
        }
        else
        {
            pointCloudMaterial.SetFloat("_PointSize", pointSize);
        }
    }

    private void InitGPUResources()
    {
        filterDepthMapKernelId = pointCloudProcessor.FindKernel("FilterDepthMap");
        generateVertexBufferKernelId = pointCloudProcessor.FindKernel("GenerateVertexBuffer");

        // Kinect1用のGPUバッファ
        inputVertexBuffer1 = new ComputeBuffer(num1, sizeof(float) * 3);
        outputVertexBuffer1 = new ComputeBuffer(num1, sizeof(float) * 3);
        validityMaskBuffer1 = new ComputeBuffer(num1, sizeof(int));
        indirectArgsBuffer1 = new ComputeBuffer(num1 + 1, sizeof(uint));
        counterBuffer1 = new ComputeBuffer(2, sizeof(uint));

        // Kinect2用のGPUバッファ
        inputVertexBuffer2 = new ComputeBuffer(num2, sizeof(float) * 3);
        outputVertexBuffer2 = new ComputeBuffer(num2, sizeof(float) * 3);
        validityMaskBuffer2 = new ComputeBuffer(num2, sizeof(int));
        indirectArgsBuffer2 = new ComputeBuffer(num2 + 1, sizeof(uint));
        counterBuffer2 = new ComputeBuffer(2, sizeof(uint));

        // データバッファ初期化
        xyzBuffer1 = new Short3[num1];
        colorBuffer1 = new BGRA[num1];
        xyzBuffer2 = new Short3[num2];
        colorBuffer2 = new BGRA[num2];
    }

    private void InitKinectDevices()
    {
        // Kinect1（デバイス0）
        kinect1 = Device.Open(0);
        kinect1.StartCameras(new DeviceConfiguration
        {
            ColorFormat = ImageFormat.ColorBGRA32,
            ColorResolution = ColorResolution.R720p,
            DepthMode = DepthMode.NFOV_2x2Binned,
            SynchronizedImagesOnly = true,
            CameraFPS = FPS.FPS30
        });
        transformation1 = kinect1.GetCalibration().CreateTransformation();

        // Kinect2（デバイス1）
        kinect2 = Device.Open(1);
        kinect2.StartCameras(new DeviceConfiguration
        {
            ColorFormat = ImageFormat.ColorBGRA32,
            ColorResolution = ColorResolution.R720p,
            DepthMode = DepthMode.NFOV_2x2Binned,
            SynchronizedImagesOnly = true,
            CameraFPS = FPS.FPS30
        });
        transformation2 = kinect2.GetCalibration().CreateTransformation();
    }

    private void InitMesh1()
    {
        width1 = kinect1.GetCalibration().DepthCameraCalibration.ResolutionWidth;
        height1 = kinect1.GetCalibration().DepthCameraCalibration.ResolutionHeight;
        num1 = width1 * height1;

        mesh1 = new Mesh();
        mesh1.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        vertices1 = new Vector3[num1];
        colors1 = new Color32[num1];

        mesh1.vertices = vertices1;
        mesh1.colors32 = colors1;

        MeshFilter mf = gameObject.GetComponent<MeshFilter>() ?? gameObject.AddComponent<MeshFilter>();
        mf.mesh = mesh1;

        MeshRenderer mr = gameObject.GetComponent<MeshRenderer>() ?? gameObject.AddComponent<MeshRenderer>();
        mr.material = pointCloudMaterial;
    }

    private void InitMesh2()
    {
        width2 = kinect2.GetCalibration().DepthCameraCalibration.ResolutionWidth;
        height2 = kinect2.GetCalibration().DepthCameraCalibration.ResolutionHeight;
        num2 = width2 * height2;

        mesh2 = new Mesh();
        mesh2.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        vertices2 = new Vector3[num2];
        colors2 = new Color32[num2];

        mesh2.vertices = vertices2;
        mesh2.colors32 = colors2;

        kinect2Object = new GameObject("Kinect2_PointCloud");
        MeshFilter mf2 = kinect2Object.AddComponent<MeshFilter>();
        mf2.mesh = mesh2;
        MeshRenderer mr2 = kinect2Object.AddComponent<MeshRenderer>();
        mr2.material = pointCloudMaterial;

        kinect2Object.transform.position = kinect2ManualPosition;
        kinect2Object.transform.eulerAngles = kinect2ManualRotation;
    }

    private void SetupExternalTransformation()
    {
        // 外部校正パラメータ（OpenCV/Open3D前提）
        float r11 = 0.8107891645868455f;
        float r12 = 0.0019960699214603833f;
        float r13 = -0.5853349009698925f;
        float r21 = -0.06550264646595835f;
        float r22 = 0.9940224675077751f;
        float r23 = -0.08734264362675243f;
        float r31 = 0.5816617005567075f;
        float r32 = 0.10915745414135877f;
        float r33 = 0.8060733938735705f;

        // 並進（ミリ→メートル換算済み）
        float t_x = -1763.7806835019967f * 0.001f;
        float t_y = -354.33494633983526f * 0.001f;
        float t_z = 1164.4870327703366f * 0.001f;

        Matrix4x4 R = Matrix4x4.identity;
        if (useTransposedRotation)
        {
            R.SetRow(0, new Vector4(r11, r21, r31, 0));
            R.SetRow(1, new Vector4(r12, r22, r32, 0));
            R.SetRow(2, new Vector4(r13, r23, r33, 0));
        }
        else
        {
            R.SetRow(0, new Vector4(r11, r12, r13, 0));
            R.SetRow(1, new Vector4(r21, r22, r23, 0));
            R.SetRow(2, new Vector4(r31, r32, r33, 0));
        }

        Matrix4x4 T_ext = Matrix4x4.identity;
        T_ext.SetRow(0, new Vector4(R[0, 0], R[0, 1], R[0, 2], t_x));
        T_ext.SetRow(1, new Vector4(R[1, 0], R[1, 1], R[1, 2], t_y));
        T_ext.SetRow(2, new Vector4(R[2, 0], R[2, 1], R[2, 2], t_z));

        // Kinect1の点群は (x, -y, z) のため、y軸反転して合わせる
        Matrix4x4 invertY = Matrix4x4.identity;
        invertY[1, 1] = -1;

        transform2To1 = invertY * T_ext;
    }

    private async Task KinectLoop1()
    {
        while (true)
        {
            using (Capture capture = await Task.Run(() => kinect1.GetCapture()).ConfigureAwait(true))
            using (Image colorImage = transformation1.ColorImageToDepthCamera(capture))
            using (Image xyzImage = transformation1.DepthImageToPointCloud(capture.Depth))
            {
                // データの取得とバッファへのコピー
                BGRA[] tempColorArray = colorImage.GetPixels<BGRA>().ToArray();
                Short3[] tempXyzArray = xyzImage.GetPixels<Short3>().ToArray();

                System.Array.Copy(tempColorArray, 0, colorBuffer1, 0, tempColorArray.Length);
                System.Array.Copy(tempXyzArray, 0, xyzBuffer1, 0, tempXyzArray.Length);

                // 点群処理
                if (useGPUAcceleration && pointCloudProcessor != null)
                {
                    ProcessPointCloudGPU1(xyzBuffer1, colorBuffer1);
                }
                else
                {
                    ProcessPointCloudCPU1(xyzBuffer1, colorBuffer1);
                }
            }
        }
    }

    private void ProcessPointCloudGPU1(Short3[] xyzArray, BGRA[] colorArray)
    {
        // 点群データ変換
        Vector3[] tempVertices = new Vector3[num1];
        for (int i = 0; i < num1; i++)
        {
            tempVertices[i] = new Vector3(
                xyzArray[i].X * 0.001f,
                -xyzArray[i].Y * 0.001f,
                xyzArray[i].Z * 0.001f
            );
            colors1[i] = new Color32(
                colorArray[i].R,
                colorArray[i].G,
                colorArray[i].B,
                255
            );
        }

        // GPU処理
        inputVertexBuffer1.SetData(tempVertices);

        uint[] counterData = new uint[2] { 0, 0 };
        counterBuffer1.SetData(counterData);

        pointCloudProcessor.SetBuffer(filterDepthMapKernelId, "inputVertices", inputVertexBuffer1);
        pointCloudProcessor.SetBuffer(filterDepthMapKernelId, "outputVertices", outputVertexBuffer1);
        pointCloudProcessor.SetBuffer(filterDepthMapKernelId, "validityMask", validityMaskBuffer1);
        pointCloudProcessor.SetBuffer(filterDepthMapKernelId, "counter", counterBuffer1);
        pointCloudProcessor.SetFloat("outlierThreshold", outlierThreshold);
        pointCloudProcessor.SetFloat("depthThreshold", depthDiscontinuityThreshold);
        pointCloudProcessor.SetInt("width", width1);
        pointCloudProcessor.SetInt("height", height1);

        pointCloudProcessor.Dispatch(filterDepthMapKernelId, Mathf.CeilToInt(width1 / 8f), Mathf.CeilToInt(height1 / 8f), 1);

        // 結果の取得とメッシュ更新
        outputVertexBuffer1.GetData(vertices1);

        UpdateMesh(mesh1, vertices1, colors1, width1, height1);
    }

    private void ProcessPointCloudCPU1(Short3[] xyzArray, BGRA[] colorArray)
    {
        for (int i = 0; i < num1; i++)
        {
            vertices1[i] = new Vector3(
                xyzArray[i].X * 0.001f,
                -xyzArray[i].Y * 0.001f,
                xyzArray[i].Z * 0.001f
            );
            colors1[i] = new Color32(
                colorArray[i].R,
                colorArray[i].G,
                colorArray[i].B,
                255
            );
        }

        if (enableNoiseFiltering)
        {
            vertices1 = FilterPointCloud(vertices1, width1, height1);
        }

        UpdateMesh(mesh1, vertices1, colors1, width1, height1);
    }

    private async Task KinectLoop2()
    {
        while (true)
        {
            using (Capture capture = await Task.Run(() => kinect2.GetCapture()).ConfigureAwait(true))
            using (Image colorImage = transformation2.ColorImageToDepthCamera(capture))
            using (Image xyzImage = transformation2.DepthImageToPointCloud(capture.Depth))
            {
                // データの取得とバッファへのコピー
                BGRA[] tempColorArray = colorImage.GetPixels<BGRA>().ToArray();
                Short3[] tempXyzArray = xyzImage.GetPixels<Short3>().ToArray();

                System.Array.Copy(tempColorArray, 0, colorBuffer2, 0, tempColorArray.Length);
                System.Array.Copy(tempXyzArray, 0, xyzBuffer2, 0, tempXyzArray.Length);

                // 点群処理
                if (useGPUAcceleration && pointCloudProcessor != null)
                {
                    ProcessPointCloudGPU2(xyzBuffer2, colorBuffer2);
                }
                else
                {
                    ProcessPointCloudCPU2(xyzBuffer2, colorBuffer2);
                }
            }
        }
    }

    private void ProcessPointCloudGPU2(Short3[] xyzArray, BGRA[] colorArray)
    {
        // 点群データ変換と外部校正適用
        Vector3[] tempVertices = new Vector3[num2];
        for (int i = 0; i < num2; i++)
        {
            Vector3 rawPoint = new Vector3(
                xyzArray[i].X * 0.001f,
                xyzArray[i].Y * 0.001f,
                xyzArray[i].Z * 0.001f
            );
            tempVertices[i] = transform2To1.MultiplyPoint3x4(rawPoint);
            colors2[i] = new Color32(
                colorArray[i].R,
                colorArray[i].G,
                colorArray[i].B,
                255
            );
        }

        // GPU処理
        inputVertexBuffer2.SetData(tempVertices);

        uint[] counterData = new uint[2] { 0, 0 };
        counterBuffer2.SetData(counterData);

        pointCloudProcessor.SetBuffer(filterDepthMapKernelId, "inputVertices", inputVertexBuffer2);
        pointCloudProcessor.SetBuffer(filterDepthMapKernelId, "outputVertices", outputVertexBuffer2);
        pointCloudProcessor.SetBuffer(filterDepthMapKernelId, "validityMask", validityMaskBuffer2);
        pointCloudProcessor.SetBuffer(filterDepthMapKernelId, "counter", counterBuffer2);
        pointCloudProcessor.SetFloat("outlierThreshold", outlierThreshold);
        pointCloudProcessor.SetFloat("depthThreshold", depthDiscontinuityThreshold);
        pointCloudProcessor.SetInt("width", width2);
        pointCloudProcessor.SetInt("height", height2);

        pointCloudProcessor.Dispatch(filterDepthMapKernelId, Mathf.CeilToInt(width2 / 8f), Mathf.CeilToInt(height2 / 8f), 1);

        // 結果の取得とメッシュ更新
        outputVertexBuffer2.GetData(vertices2);

        UpdateMesh(mesh2, vertices2, colors2, width2, height2);
    }

    private void ProcessPointCloudCPU2(Short3[] xyzArray, BGRA[] colorArray)
    {
        for (int i = 0; i < num2; i++)
        {
            Vector3 rawPoint = new Vector3(
                xyzArray[i].X * 0.001f,
                xyzArray[i].Y * 0.001f,
                xyzArray[i].Z * 0.001f
            );
            vertices2[i] = transform2To1.MultiplyPoint3x4(rawPoint);
            colors2[i] = new Color32(
                colorArray[i].R,
                colorArray[i].G,
                colorArray[i].B,
                255
            );
        }

        if (enableNoiseFiltering)
        {
            vertices2 = FilterPointCloud(vertices2, width2, height2);
        }

        UpdateMesh(mesh2, vertices2, colors2, width2, height2);
    }

    // メッシュの更新処理を統合
    private void UpdateMesh(Mesh mesh, Vector3[] vertices, Color32[] colors, int width, int height)
    {
        mesh.vertices = vertices;
        mesh.colors32 = colors;
        List<int> indiceList = GetIndiceList(vertices, width, height, meshTopology);
        mesh.SetIndices(indiceList, meshTopology, 0);
        mesh.RecalculateBounds();
    }

    void Update()
    {
        // FPS計測
        frameCount++;
        frameTime += Time.deltaTime;

        if (frameTime >= 1.0f)
        {
            fps = frameCount / frameTime;
            frameCount = 0;
            frameTime = 0;
        }
    }

    void OnGUI()
    {
        // パフォーマンス表示とモード切替
        string mode = useGPUAcceleration ? "GPU Mode" : "CPU Mode";
        GUI.Label(new Rect(10, 10, 200, 20), $"FPS: {fps:F1} - {mode}");

        if (pointCloudProcessor != null && GUI.Button(new Rect(10, 40, 150, 30), "Switch GPU/CPU"))
        {
            useGPUAcceleration = !useGPUAcceleration;
        }
    }

    private void OnDestroy()
    {
        // Kinectリソース解放
        if (kinect1 != null)
        {
            kinect1.StopCameras();
            kinect1.Dispose();
        }
        if (kinect2 != null)
        {
            kinect2.StopCameras();
            kinect2.Dispose();
        }

        // GPUリソース解放
        ReleaseComputeBuffer(ref inputVertexBuffer1);
        ReleaseComputeBuffer(ref outputVertexBuffer1);
        ReleaseComputeBuffer(ref validityMaskBuffer1);
        ReleaseComputeBuffer(ref indirectArgsBuffer1);
        ReleaseComputeBuffer(ref counterBuffer1);

        ReleaseComputeBuffer(ref inputVertexBuffer2);
        ReleaseComputeBuffer(ref outputVertexBuffer2);
        ReleaseComputeBuffer(ref validityMaskBuffer2);
        ReleaseComputeBuffer(ref indirectArgsBuffer2);
        ReleaseComputeBuffer(ref counterBuffer2);
    }

    private void ReleaseComputeBuffer(ref ComputeBuffer buffer)
    {
        if (buffer != null)
        {
            buffer.Release();
            buffer = null;
        }
    }

    private List<int> GetIndiceList(Vector3[] vertices, int pointWidth, int pointHeight, MeshTopology topology)
    {
        List<int> indiceList = new List<int>();

        if (topology == MeshTopology.Points)
        {
            // 各頂点のインデックス追加（有効点のみ）
            for (int i = 0; i < vertices.Length; i++)
            {
                if (vertices[i].magnitude != 0)
                {
                    indiceList.Add(i);
                }
            }
            return indiceList;
        }

        float depthThreshold = depthDiscontinuityThreshold;

        // グリッド状のメッシュ生成
        for (int y = 0; y < pointHeight - 1; y++)
        {
            for (int x = 0; x < pointWidth - 1; x++)
            {
                int index = y * pointWidth + x;
                int a = index;
                int b = index + 1;
                int c = index + pointWidth;
                int d = index + pointWidth + 1;

                bool validA = vertices[a].magnitude != 0;
                bool validB = vertices[b].magnitude != 0;
                bool validC = vertices[c].magnitude != 0;
                bool validD = vertices[d].magnitude != 0;

                // 深度不連続性の検出
                bool depthDiscontinuityAB = validA && validB && Mathf.Abs(vertices[a].z - vertices[b].z) > depthThreshold;
                bool depthDiscontinuityAC = validA && validC && Mathf.Abs(vertices[a].z - vertices[c].z) > depthThreshold;
                bool depthDiscontinuityBD = validB && validD && Mathf.Abs(vertices[b].z - vertices[d].z) > depthThreshold;
                bool depthDiscontinuityCDbool = validC && validD && Mathf.Abs(vertices[c].z - vertices[d].z) > depthThreshold;

                bool canConnectABC = validA && validB && validC && !depthDiscontinuityAB && !depthDiscontinuityAC;
                bool canConnectBCD = validB && validC && validD && !depthDiscontinuityBD && !depthDiscontinuityCDbool;
                bool canConnectABCD = canConnectABC && canConnectBCD;

                switch (topology)
                {
                    case MeshTopology.Triangles:
                        if (canConnectABC)
                        {
                            indiceList.Add(a);
                            indiceList.Add(b);
                            indiceList.Add(c);
                        }
                        if (canConnectBCD)
                        {
                            indiceList.Add(c);
                            indiceList.Add(b);
                            indiceList.Add(d);
                        }
                        break;
                    case MeshTopology.Quads:
                        if (canConnectABCD)
                        {
                            indiceList.Add(a);
                            indiceList.Add(b);
                            indiceList.Add(d);
                            indiceList.Add(c);
                        }
                        break;
                    default:
                        // Lines系の処理
                        if (validA && validB && !depthDiscontinuityAB)
                        {
                            indiceList.Add(a);
                            indiceList.Add(b);
                        }
                        if (validC && validD && !depthDiscontinuityCDbool)
                        {
                            indiceList.Add(c);
                            indiceList.Add(d);
                        }
                        break;
                }
            }
        }
        return indiceList;
    }

    private Vector3[] FilterPointCloud(Vector3[] vertices, int width, int height)
    {
        Vector3[] filtered = new Vector3[vertices.Length];
        System.Array.Copy(vertices, filtered, vertices.Length);

        int windowSize = 3;
        int halfWindow = windowSize / 2;

        // メディアンフィルタによるノイズ除去
        for (int y = halfWindow; y < height - halfWindow; y++)
        {
            for (int x = halfWindow; x < width - halfWindow; x++)
            {
                int centerIndex = y * width + x;

                if (vertices[centerIndex].magnitude == 0)
                    continue;

                List<float> depths = new List<float>();
                for (int dy = -halfWindow; dy <= halfWindow; dy++)
                {
                    for (int dx = -halfWindow; dx <= halfWindow; dx++)
                    {
                        int idx = (y + dy) * width + (x + dx);
                        if (vertices[idx].magnitude != 0)
                        {
                            depths.Add(vertices[idx].z);
                        }
                    }
                }

                if (depths.Count < 4)
                    continue;

                depths.Sort();
                float medianDepth = depths[depths.Count / 2];

                if (Mathf.Abs(vertices[centerIndex].z - medianDepth) > outlierThreshold)
                {
                    filtered[centerIndex] = Vector3.zero;
                }
            }
        }

        return filtered;
    }
}
