import socket
import campo
import time
import os

# setando conexão
sock_dados = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

ip_destino = '127.0.0.1'
porta_destino = 50400
destino = (ip_destino, porta_destino)

sock_dados.connect(destino)

jogador = campo.cenario()
adversario = campo.cenario()

os.system('cls' if os.name == 'nt' else 'clear')
print('---- BATALHA NAVAL ----')
print('Selecione as posições do seu navio por linha coluna e direção (B=Baixo, D=Direita). Exemplo: C3B')

print('\n---- VOCE -----')
jogador.exibe()

for i in range(jogador.nr_barcos):
    st_sucesso = -1
    while st_sucesso == -1:
        navio = input('Selecione a posição do Navio '+str(i+1)+': ')
        
        try:
            dir = 'rigth'
            if len(navio) == 3 and navio[2] == 'B':
                dir = 'down'

            st_sucesso = jogador.set_navio(navio[0],navio[1],i+1,dir)

            if st_sucesso == -1:
                print('Posição e ou direção inválida para o navio, insira novamente.')
        except:
            print('Posição e ou direção inválida para o navio, insira novamente.')

    os.system('cls' if os.name == 'nt' else 'clear')
    print('---- BATALHA NAVAL ----')
    print('Selecione as posições do seu navio por linha coluna e direção (B=Baixo, D=Direita). Exemplo: C3B')

    print('\n---- VOCE -----')
    jogador.exibe()

print('Aguardando o adversário inserir os navios.')

ok_serv = sock_dados.recv(1)

ok_client = 1
ok_client_bytes = ok_client.to_bytes(1,byteorder='big',signed=True)
sock_dados.send(ok_client_bytes)

print('Campo de batalha definido. Começando o jogo.')
time.sleep(2)

#limpa a tela
os.system('cls' if os.name == 'nt' else 'clear')

finalizado = False

def atualiza_tela():
    os.system('cls' if os.name == 'nt' else 'clear')
    print('---- VOCE -----')
    jogador.exibe()
    print('\n-- ADVERSARIO --')
    adversario.exibe()
    print('\n1,2,3 = Navio | * = Tiro na agua | @ = Tiro no navio | . = Desconhecido.')

ganhou = False

while not finalizado:
    atualiza_tela()
    # dando um tiro
    st_sucesso = -1
    while st_sucesso == -1:
        try:
            
            tiro = input('Selecione a posição linha coluna para dar um tiro: ')
            
            px = ord(tiro[0]) - 65 #convertendo de letra para a respectiva linha
            py = int(tiro[1]) - 1
            
            # enviando dados
            comando_bytes = tiro.encode(encoding='UTF-8')
            
            sock_dados.send(comando_bytes)
            
            tipo_bytes = sock_dados.recv(2)
            
            st_sucesso = int.from_bytes(tipo_bytes, byteorder='big', signed=True)
            
            if st_sucesso == -3:
                print('Posição '+tiro+' já foi atirada. Atire novamente.')
                st_sucesso = -1
            elif st_sucesso == -1:
                print('Erro ao atirar na posição '+tiro+'. Atire novamente.')
        except:
            st_sucesso = -1

    px = ord(tiro[0]) - 65 #convertendo de letra para a respectiva linha
    py = int(tiro[1]) - 1

    if st_sucesso == 0:
        adversario.campo[px][py] = '*'
    elif st_sucesso == -2:
        adversario.campo[px][py] = '@'
        finalizado = True
        ganhou = True
        break
    else:
        adversario.campo[px][py] = '@'
    
    atualiza_tela()
    print('Aguardando tiro do adversário..')

    #recebendo um tiro
    st_sucesso = -1
    while st_sucesso == -1:
        tiro_adversario = sock_dados.recv(2)
        shot = tiro_adversario.decode(encoding='UTF-8')
        #print(shot)

        st_sucesso = jogador.retorna_tiro(shot[0],shot[1])
        
        # enviando dados
        mensagem = st_sucesso.to_bytes(2, byteorder='big', signed=True)
        
        sock_dados.send(mensagem)

        #controle do loop fica só pelo fator -1
        if st_sucesso == -3:
            st_sucesso = -1
            
        # Caso tenha perdido o jogo
        if st_sucesso == -2:
            ganhou = False
            finalizado = True


atualiza_tela()
if ganhou:
    print('----  Você GANHOU :)  ----')
else:
    print('----  Você PERDEU :/  ----')

input('Pressiona qualquer tecla para sair...')

sock_dados.close()