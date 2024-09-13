
export const getAIMessage = async (userQuery) => {


  try{
    const response = await fetch("http://127.0.0.1:5000/get-ai-message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: userQuery }),

    });

    const message = await response.json();

    
    let content = "";
    if (message.content) {
      content += `<p>${message.content}</p>`;
    }
    if (message.image_url) {
      content += `<img src="${message.image_url}" alt="Generated Graph" style="max-width: 100%; height: auto;" />`;
    }

    return {
      role: "assistant",
      content: content,
    };
    

    } catch (error){
      return { role: "assistant", content: "Can you rephrase your question?" };
    }
  
};
