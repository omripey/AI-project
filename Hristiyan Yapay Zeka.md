Sorun:
→Ortada muhabbet edilebilecek, sorular sorulup cevaplar alınabilecek, gerekirse kaynak verebilecek yapay zekalar çokça mevcut. Ancak, bu yapay zekalar seküler alanlarda başarılıyken Hristiyanlık konusunda istenen dilde veya formatta cevaplar veremiyor.

Çözüm:
→Hristiyan kelime dağarcığına aşina olan ve Hristiyan kaynaklara erişimi olan bir yapay zeka alternatifi geliştirmek.

Süreç:
→Öncelikle GPT ve muadilleriyle başlamayı planladım ama bu modeller bizim amacımız için yanlış yönde eğitildiler, bu yüzden daha basit ve “düşük zekalı” ama bir yandan da maliyeti ucuz bir model olan Phi-2’ye yöneldim ve bütün sistemimi bu model üstüne kurdum.

Başlangıç:
→Başta herhangi bir eğitim ve kaynak referansı olmadan verilen bir dini soruya cevap verir gibi yapıp düzenli olarak farazi durumlar kuruyordu-buna yapay zeka halüsinasyonu deniyor-ve önüne geçilmezse konuşup konuşup duruyor, örneğin kutsal kitapta sevgi ile ilgili ne deniyor sorusuna paradan başlayıp varsayımsal bir biyokimya laboratuvarında insanların arasındaki dinamikler hakkında gevelemeye başladı.

Bunun önüne geçmek için 2 öncü sistem kurdum:
→Birincisi system_prompt denen yapay zekaya nasıl davranmasını söyleyen cümlelerin oluşturduğu bir array. Birkaç örnek vermek gerekirse:
“Her cevabın doğal ve kutsal kitaba uygun olacak.”
“Farazi hikayeler, karakterler, senaryolar oluşturma. Sana ne sorulursa ona cevap ver.”
“Sorulara bulduğun ayetleri ‘Cümle. (Kitap Bölüm No.: Ayet No.)’ formatında kur.”
Bu system_prompt array’ine eklenen cümleler yapay zekanın kişiliğini, nasıl konuşacağını, davranışlarını kontrol etmek konusunda elzem. Şu an 14 cümle mevcut, ama bu cümleler ileride çeşitlendirilip daha etkili yazılmalı, çünkü istediğimiz şekle sokmanın ana yollarından biri bu array’i zenginleştirmek.
→İkincisi de clean_answer adlı kurduğum fonksiyon. Phi-2 katı kısıtlamalar konulmadığı sürece bahsedilen halüsinasyonlara kapıldığı için “Varsay ki”, “Örneğin”, “Kuramsal olarak”, “Şu durumda olduğunu düşün”, “Düşünce egzersizi” gibi sözlerle konudan dağılabiliyor. Bunu farazi konuşmada kullanılan birçok başlangıç kelimesini gördüğü anda kelime/kelimeleri keserek, basitçe söylemek gerekirse nokta görene kadar hem harflerden hem boşluklardan kurtuluyor. Başta modelden gelen cevapta 3-4 anlamlı cümleden sonra “Bir örnek düşünelim” diyip oradan sonrasına devam etmiyordu, ama artık tutarlı olarak anlamlı ve varsayımsal cümlelerden arınmış bir cevap verebiliyor.

Bu iki sistemden sonra kendim kutsal kitabı yükleyip, ayetleri ayrıştırıp anahtar kelimeler aramaya karar verdim. Başta açıklamalı ve yorumlamalı, alışkın olduğumuz kutsal kitap formatında yükledim, ancak bu formatta dipnotlar ve özel karakterler yüzünden algoritmanın kafası karışıyordu. Bunun için ortaya çıkabilecek bütün özel karakterleri ayrıştırarak “parsed and deduped”, yani yapay zekanın daha kolay okuyabileceği bir formata dönüştürdüm.

Bu sistem ile “para”, “sevgi”, “günah”, “nefret”, “öfke” gibi akla ilk gelen kavramları bir array içine koyup, kullanıcının sorduğu soru içinde bu array’deki kelimeleri ve çoğullarını aratmayı denedim. Buna hardcoded fix dendiğini öğrendim; dinamik bir sistem, yani duruma göre adapte olamayan bir sistem olmamakla birlikte, listedeki kelimeleri bulamadığında tekrar halüsinasyon moduna girip konudan sapıyordu. Kullanıcının sorabilecekleri bütün potansiyel sorular için teker teker kelimeler seçmenin zaman kaybı olduğuna karar verdim, ve bu işin daha otomatik bir şekilde yapılabileceğini düşündüm.

→ Bu noktada kelime aramayı otomatikleştirmek için kendi tokenizer yapımı kurdum. Özetle, kullanıcıdan gelen soruyu token’lara (kelime birimlerine) ayırıp, bu token’ları daha önce ayıklanmış ve temizlenmiş ayet metinleri üzerinde taramak üzere bir yapı kurdum. Bu yapı her kelime için, ayet içinde geçen tüm varyasyonlarını (çoğul, çekim eki, büyük/küçük harf farkı vb.) normalize ederek arama yapıyor.

Phi-2:
Microsoft Research tarafından yapılan Causal Language modelini kullanan bir yapay zeka-hem anlamlı hem de önceki bağlamla örtüşen yanıtlar verebilen, “masked self-attention” sistemiyle kronolojik mesaj yaratımı. Daha basit söylemek gerekirse, bir cümle kurarken önceki kelimelere bakarak sonraki kelimeyi tahmin ediyor; hiçbir zaman ileriye değil, geriye bakarak mesaj üretiyor. Bu sistem GPT modelinde de kullanılıyor, Phi-2 bu açıdan daha “cost effective” bir versiyon. 2.7 milyar parametresiyle muadillerine nazaran çok daha küçük bir model olmasına rağmen, yaratıcılarının “better data beats more data” prensibiyle kendisine verilen ders kitabına benzer formatıyla hızlı, ucuz, yüksek performans sergileyebiliyor. Şaşırtıcı şekilde, piyasadaki muadillerinin üçte biri büyüklüğe sahip olmasına rağmen benchmark testlerinde üstün performans sergiliyor. LLaMa, Mistral, GPT gibi modellerle karşılaştırıldığında kaynak başına hızı bu modelleri solda sıfır bırakıyor. Tek “dezavantajı” pretuned ama instruction tuned değil, yani basit eğitimlere sahip olmasına rağmen özelleştirilmiş değil. Bu kesinlikle bizim projemiz için büyük bir avantaj çünkü kolayca eğilebilen yaş ağaç gibi kendimiz istediğimiz şekle sokabiliriz. 2048 token sınırı düşük, güvenlik parametreleri üstüne çalışılması gerek, ama bu “insincere” yapay zeka durumundan kurtulmak için üstüne kafa yorulup zaman verilip dilenen formata varılabilir.

FAISS (Facebook AI Similarity Search):
Yüzlerce/binlerce kelime yazıp her prompt için o büyük listede gezinmek yerine prompt içindeki anahtar kelimeleri veya bu kelimelere benzer anlamlı kelimeleri kutsal kitap ayetleri içinde aramaya karar verdim, bunun için karşıma çıkan algoritma da FAISS oldu. Bu modeli bir embedding (gömme) modeli ile birlikte kullanınca spesifik kelimeler değil, verilen bütün cümlenin taşıdığı anlam göz önünde bulundurulduğu için ortaya çıkan sonuçlar çok daha geniş bir kapsam içinde oluyor.
Teknik konuya inmek gerekirse, “What does the Bible say about marriage?” gibi bir örnek cümlede her kelimeye bir benzerlik numarası veriliyor. e5-large-v2 modeli ile yapılan bu gömme işlemi cümleyi bir vektör haline getirerek semantik anlam aramasına hazırlıyor. Bundan sonra FAISS geniş veritabanı içinde bunla bitişik cümleleri analiz ediyor. Örneğin, “Is it a sin for someone to separate from their spouse?” ve “Is the end of a marriage frowned upon in the Bible?” gibi sorular bu vektör alanında birbirine yakın oturuyorlar. Bundan sonra, kosinüs benzerliği denen vektörler arasındaki yakınlığın nümerik sayılarını analiz ederek bir eşik değerin üstündeki benzerlikte olan ayetleri getiriyor. Bu ayetlerin benzerlik değeri en yüksek olan üçü kullanıcıya verilen yanıttaki “Biblical Teaching” kısmında yapay zeka tarafından açıklanıyor, altında da “Relevant Verses” kısmına referans olarak konuluyor.

→ FAISS aramaları, yalnızca kullanıcıdan gelen soruya en benzer ayetleri değil, aynı zamanda aralarında bağlam tutarlılığı olan ayetleri de getiriyor. Bu bağlam ilişkisini sağlamak için arka planda benzerlik eşiği dışında bir minimum bağlam uzunluğu da kontrol ediliyor.

Paraphrase-multilingual-mpnet-base-v2:
Kutsal Kitap ayetleri ile kullanıcının soruları arasındaki anlamsal benzerliği ölçmek için, paraphrase-multilingual-mpnet-base-v2 adlı çok dilli ve paraphrase (anlamdaş cümle) odaklı bir embedding modeli kullandım. Bu model, verilen cümleleri çok boyutlu vektörlere dönüştürerek, farklı dillerde olsa bile aynı anlama gelen cümlelerin birbirine yakın konumlanmasını sağlıyor. Böylece, örneğin “Evlilik hakkında ne diyor?” ve “Birinin eşiyle ayrılması günah mıdır?” gibi farklı şekillerde sorulmuş sorular aynı anlam alanında bulunuyor ve ilgili ayetler kolayca bulunabiliyor.
Bu model Microsoft ve SentenceTransformers tarafından geliştirilen MPNet mimarisini temel alır ve özellikle anlamlı, bağlama uygun cümle temsilinde üstündür. Ayrıca çok dilli destek sunduğu için ileride farklı dilde sorularla da çalışabilir. Bu embeddingler FAISS arama motoru ile birlikte kullanılarak, kullanıcı sorusuna en uygun ayetleri hızlı ve etkili şekilde bulmamı sağladı.

e5-large-v2:
NLP alanında en etkili modellerden biri, ana amacı hem bilgisayarlara hem de insanlara önlerindeki yazıyı anlamdırmak. Barındırdığı 24 katman ve 1024 karakteri gömme sınırı ile yazılar arasındaki ilişkiyi ve anlamlarını karşılaştırmak için biçilmiş kaftan. Üstte açıklandığı gibi, cümledeki kelimeleri semantik anlamlarına göre teker teker sayılara çevirerek cümleyi bir vektör haline getiriyor. Bu cümlede sayılar kendi başına bir anlam ifade etmiyor ancak bu vektör bir “meaning space” içine yerleştirildiğinde benzer cümleler birbirine yakın oluyor. Kurduğum yapay zeka modelinin kullandığı sistem de bu. Yüklenen kutsal kitaptaki ayetleri bu vektörlere ayrıştırarak verilen prompt cümlesiyle “meaning space” içinde en yakın olan ayetleri çekerek benzerlik sayısı en yüksek 3 ayetle bir yanıt oluşturuyor.

→ Bu modeli yalnızca FAISS ile kullanmadım. Aynı zamanda kelime başına düşen embedding uzaklıklarının istatistiksel dağılımını analiz eden bir helper fonksiyon da oluşturdum. Bu sayede bazı ayetlerin, anlam olarak soruyla uyuşsa da genel dilinden dolayı zayıf sonuçlar getirmesi engellenmiş oldu.

→Burada şunu da eklemek lazım, Phi-2 modelini sadece yanıt oluşturmak için kullanıyorum; asıl anlam benzerliği karşılaştırmaları FAISS ve e5-large-v2 ile gerçekleşiyor. Phi-2 bu anlam eşleştirmeleriyle gelen ayetleri, kendi doğal dil yetenekleriyle sade, özlü ve sadık bir şekilde açıklıyor.

Bundan sonra istediğim altyapıyı kurduktan sonra ayet referansı görevini basitleştirmeye karar verdim. Şu ana kadar kullandığım kutsal kitap açıklamalar, yorumlar ve dipnotlarla doluydu. Bu ekstralardan kurtulmak için bir ayıklama algoritması yapmıştım ancak modelimin işini kolaylaştırmak için daha basit bir formatla yapılmış kutsal kitapları aramaya başladım. Sonunda, bütün kitabın aşağıdaki formatta olduğu bir versiyonla karşılaştım:

Genesis
[1:1] In the beginning God created the heaven and the earth.
[1:2] And the earth was without form and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.
[1:3] And God said, Let there be light: and there was light.
Modelin kitabı inceleme algoritmasını buna göre değiştirdim(eski kısmı silmeden yorum yaparak) ve bu işi daha da akıcı hale getirdi.

Ayet Tamamlama Fonksiyonu: get_extended_verse()
Model, genellikle doğru cevaplar üretse de zaman zaman verdiği ayetler eksik olabiliyordu. Bunun temel nedeni, Kutsal Kitap'taki bazı cümlelerin tek bir sonlandırıcı işaret (., !, ?) yerine noktalı virgül (;), iki nokta (:) veya virgül (,) ile devam etmesi ve bu nedenle cümlelerin birden fazla ayete bölünmüş olmasıydı.

Örneğin:
For though there be that are called gods, whether in heaven or in earth, (as there be gods many, and lords many,)
(I Corinthians 8:5)
But to us there is but one God, the Father...
(I Corinthians 8:6)

Model yalnızca 8:5 ayetini verirken cümle tamamlanmadığı için bağlam eksik görünüyordu.

Bu durumu düzeltmek için get_extended_verse() fonksiyonunu yazdım. Bu fonksiyon, Relevant Bible Verses bloğundaki tüm ayetlerin son karakterini kontrol eder. Eğer ayet :, ;, , gibi bir geçici duraklama işaretiyle bitiyorsa, sonraki ayet(ler)i de eklemeye devam eder. Bu işlem, ayetin sonunda gerçek bir cümle sonlandırıcısı (., !, ?) gelene kadar sürer. Ancak, bazı durumlarda bir cümle 4–5 ayete kadar uzayabiliyor. Bu durumda paragrafın fazla uzamasını engellemek ve cevabın okunabilirliğini korumak için maksimum 3 ayetten sonra cümle hâlâ tamamlanmamış olsa bile durdurma mekanizması ekledim. Bu vesileyle, birkaç ayete uzanan cümleler Relevant Bible Verses başlığı altında anlam bütünlüğünü koruyarak gösteriliyor.

Örnek Soru: “Are there many gods?”
Cevapta kullanılan ayet: I Corinthians 8:5
get_extended_verse çıktısı: [I Corinthians 8:5, I Corinthians 8:6]

Gradio Arayüzü
Şimdilik tam istenen şekilde çalışmasa bile, Gradio adlı grafik arayüz programını kullanarak basit bir internet sitesi formatı yarattım. Bu basit sistemde soruların yazıldığı boşluk, submit tuşu, ve daha geliştirilmemiş "bu cevabı sevdim" ve "bu cevabı sevmedim" şeklinde yapay zekaya feedback veren iki tuş daha ekledim. 
