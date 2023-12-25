
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001608: BF89FC07
	s_load_b64 s[2:3], s[2:3], null                            // 00000000160C: F4040081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_max_i32 s2, s3, s2                                       // 000000001618: 8A020203
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000161C: BF870009
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s2              // 000000001620: CA100080 00000002
	global_store_b32 v0, v1, s[0:1]                            // 000000001628: DC6A0000 00000100
	s_nop 0                                                    // 000000001630: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001634: BFB60003
	s_endpgm                                                   // 000000001638: BFB00000
