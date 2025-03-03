using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.Azure.Kinect.Sensor;
using System.Threading.Tasks;
using System.Linq;

public class MultiKinectMeshScript : MonoBehaviour
{
    // 共通マテリアル（Inspectorで設定可能）
    public Material pointCloudMaterial;

    // 回転行列の扱い（true: 転置版、false: JSONにあるまま）
    public bool useTransposedRotation = true;

    // 点のサイズ設定（Inspectorで調整可能）
    public float pointSize = 5.0f;

    // メッシュ品質の調整パラメータ（Inspectorで調整可能）
    [Header("メッシュ品質の設定")]
    [Tooltip("深度の不連続性を検出するしきい値。大きくすると接続が増える")]
    public float depthDiscontinuityThreshold = 0.05f;
    [Tooltip("ノイズ除去のフィルタを有効にする")]
    public bool enableNoiseFiltering = true;
    [Tooltip("中央値からの外れ値を検出するしきい値。大きくするとノイズ除去が弱くなる")]
    public float outlierThreshold = 0.1f;

    // Kinect2の手動調整パラメータ（Inspectorで調整可能）
    public Vector3 kinect2ManualPosition = new Vector3(-0.39f, 0.11f, -1.13f);
    public Vector3 kinect2ManualRotation = new Vector3(0.167f, -0.801f, -4.548f);

    // 表示するMeshのTopology（Points/Triangles/Quadsなど）
    public MeshTopology meshTopology = MeshTopology.Points;

    // Kinectデバイス
    private Device kinect1;
    private Device kinect2;

    // Kinect1用Meshとデータ
    private Mesh mesh1;
    private Vector3[] vertices1;
    private Color32[] colors1;
    private int num1;
    private int width1;
    private int height1;

    // Kinect2用Meshとデータ
    private Mesh mesh2;
    private Vector3[] vertices2;
    private Color32[] colors2;
    private int num2;
    private int width2;
    private int height2;

    // 各KinectのTransformation
    private Transformation transformation1;
    private Transformation transformation2;

    // Kinect2→Kinect1への外部校正パラメータ（JSONから取得したパラメータのUnity用変換行列）
    private Matrix4x4 transform2To1;

    // Kinect2用Mesh表示オブジェクト
    private GameObject kinect2Object;

    void Start()
    {
        // マテリアルが未設定なら作成
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
                Debug.LogError("Custom/ColoredVertex シェーダーが見つかりません。Assets/ColoredVertex.shaderを確認してください。");
                shader = Shader.Find("Standard");
                if (shader != null)
                {
                    Debug.LogWarning("フォールバックとしてStandardシェーダーを使用します。");
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

        // デバイス初期化
        InitKinectDevices();

        // Mesh初期化
        InitMesh1();
        InitMesh2();

        // 外部校正パラメータから変換行列セットアップ
        SetupExternalTransformation();

        // 非同期に各デバイスの点群更新ループを開始
        Task t1 = KinectLoop1();
        Task t2 = KinectLoop2();
    }

    // Kinectの初期化（両デバイス）
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

    // Kinect1用Meshの初期化（このスクリプトがアタッチされたGameObjectに設定）
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
        // 初期段階ではインデックスは後で更新する

        MeshFilter mf = gameObject.GetComponent<MeshFilter>();
        if (mf == null)
        {
            mf = gameObject.AddComponent<MeshFilter>();
        }
        mf.mesh = mesh1;

        MeshRenderer mr = gameObject.GetComponent<MeshRenderer>();
        if (mr == null)
        {
            mr = gameObject.AddComponent<MeshRenderer>();
        }
        mr.material = pointCloudMaterial;
    }

    // Kinect2用Meshの初期化（新規GameObjectを生成）
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
        // インデックスは更新ループ内で計算

        kinect2Object = new GameObject("Kinect2_PointCloud");
        MeshFilter mf2 = kinect2Object.AddComponent<MeshFilter>();
        mf2.mesh = mesh2;
        MeshRenderer mr2 = kinect2Object.AddComponent<MeshRenderer>();
        mr2.material = pointCloudMaterial;

        // 手動調整した位置・回転を適用
        kinect2Object.transform.position = kinect2ManualPosition;
        kinect2Object.transform.eulerAngles = kinect2ManualRotation;

        Debug.Log($"Kinect2_PointCloudに手動調整を適用: 位置({kinect2ManualPosition}), 回転({kinect2ManualRotation})");
    }

    // 外部校正パラメータからUnity用変換行列をセットアップ
    private void SetupExternalTransformation()
    {
        // JSONから取得した外部校正パラメータ（OpenCV/Open3D前提）
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

        // Kinect1の点群は (x, -y, z) のため、Kinect2のraw点群にT_ext適用後、y軸反転して合わせる
        Matrix4x4 invertY = Matrix4x4.identity;
        invertY[1, 1] = -1;

        transform2To1 = invertY * T_ext;
    }

    // Kinect1の点群取得・更新ループ
    private async Task KinectLoop1()
    {
        while (true)
        {
            using (Capture capture = await Task.Run(() => kinect1.GetCapture()).ConfigureAwait(true))
            {
                // カラー画像をDepthカメラに合わせる
                Image colorImage = transformation1.ColorImageToDepthCamera(capture);
                BGRA[] colorArray = colorImage.GetPixels<BGRA>().ToArray();

                // Depth画像から点群を取得（(x, y, z)）
                Image xyzImage = transformation1.DepthImageToPointCloud(capture.Depth);
                Short3[] xyzArray = xyzImage.GetPixels<Short3>().ToArray();

                for (int i = 0; i < num1; i++)
                {
                    // Kinect1は (x, -y, z) に変換
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

                // 点群をフィルタリングしてノイズを除去
                vertices1 = FilterPointCloud(vertices1, width1, height1);

                // 頂点・カラー更新
                mesh1.vertices = vertices1;
                mesh1.colors32 = colors1;
                // インデックスリストの再構築
                List<int> indiceList = GetIndiceList(vertices1, width1, height1, meshTopology);
                mesh1.SetIndices(indiceList, meshTopology, 0);
                mesh1.RecalculateBounds();
            }
        }
    }

    // Kinect2の点群取得・更新ループ（外部校正変換適用）
    private async Task KinectLoop2()
    {
        while (true)
        {
            using (Capture capture = await Task.Run(() => kinect2.GetCapture()).ConfigureAwait(true))
            {
                Image colorImage = transformation2.ColorImageToDepthCamera(capture);
                BGRA[] colorArray = colorImage.GetPixels<BGRA>().ToArray();

                Image xyzImage = transformation2.DepthImageToPointCloud(capture.Depth);
                Short3[] xyzArray = xyzImage.GetPixels<Short3>().ToArray();

                for (int i = 0; i < num2; i++)
                {
                    // Kinect2のraw点（ミリ→メートル換算のみ）
                    Vector3 rawPoint = new Vector3(
                        xyzArray[i].X * 0.001f,
                        xyzArray[i].Y * 0.001f,
                        xyzArray[i].Z * 0.001f
                    );
                    // 外部校正変換を適用し、Kinect1と同じくy軸反転して合わせる
                    vertices2[i] = transform2To1.MultiplyPoint3x4(rawPoint);
                    colors2[i] = new Color32(
                        colorArray[i].R,
                        colorArray[i].G,
                        colorArray[i].B,
                        255
                    );
                }

                // 点群をフィルタリングしてノイズを除去
                vertices2 = FilterPointCloud(vertices2, width2, height2);

                mesh2.vertices = vertices2;
                mesh2.colors32 = colors2;
                List<int> indiceList = GetIndiceList(vertices2, width2, height2, meshTopology);
                mesh2.SetIndices(indiceList, meshTopology, 0);
                mesh2.RecalculateBounds();
            }
        }
    }

    // Kinect終了時のリソース解放
    private void OnDestroy()
    {
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
    }

    // 頂点配列から、指定のMeshTopologyに応じたindicesリストを生成するメソッド
    private List<int> GetIndiceList(Vector3[] vertices, int pointWidth, int pointHeight, MeshTopology topology)
    {
        List<int> indiceList = new List<int>();

        if (topology == MeshTopology.Points)
        {
            // 各頂点が有効（非ゼロ）ならインデックスに追加
            for (int i = 0; i < vertices.Length; i++)
            {
                if (vertices[i].magnitude != 0)
                {
                    indiceList.Add(i);
                }
            }
            return indiceList;
        }

        // 深度の急激な変化を検出するためのしきい値
        float depthThreshold = depthDiscontinuityThreshold; // インスペクターで設定した値を使用

        // グリッド状のメッシュとして、各セルごとにインデックスを作成
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

                // 各点間の深度差を計算
                bool depthDiscontinuityAB = validA && validB && Mathf.Abs(vertices[a].z - vertices[b].z) > depthThreshold;
                bool depthDiscontinuityAC = validA && validC && Mathf.Abs(vertices[a].z - vertices[c].z) > depthThreshold;
                bool depthDiscontinuityBD = validB && validD && Mathf.Abs(vertices[b].z - vertices[d].z) > depthThreshold;
                bool depthDiscontinuityCDbool = validC && validD && Mathf.Abs(vertices[c].z - vertices[d].z) > depthThreshold;

                // 深度の不連続性がある場合は接続しない
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
                        // その他、Line系など
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

    // 点群のフィルタリングを行い、ノイズを減らす
    private Vector3[] FilterPointCloud(Vector3[] vertices, int width, int height)
    {
        // フィルタリングが無効の場合はそのまま返す
        if (!enableNoiseFiltering)
        {
            return vertices;
        }

        Vector3[] filtered = new Vector3[vertices.Length];
        System.Array.Copy(vertices, filtered, vertices.Length);

        // メディアンフィルタの窓サイズ
        int windowSize = 3;
        int halfWindow = windowSize / 2;

        for (int y = halfWindow; y < height - halfWindow; y++)
        {
            for (int x = halfWindow; x < width - halfWindow; x++)
            {
                int centerIndex = y * width + x;

                // 中心点が無効なら処理しない
                if (vertices[centerIndex].magnitude == 0)
                    continue;

                // 近傍点の深度値を収集
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

                // 有効な深度値が少なすぎる場合はスキップ
                if (depths.Count < 4)
                    continue;

                // 外れ値の除去（深度値の中央値から大きく外れる点を無視）
                depths.Sort();
                float medianDepth = depths[depths.Count / 2];

                // 中央値から大きく外れる場合は点を無効化
                if (Mathf.Abs(vertices[centerIndex].z - medianDepth) > outlierThreshold)
                {
                    filtered[centerIndex] = Vector3.zero;
                }
            }
        }

        return filtered;
    }
}
