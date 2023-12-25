
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_27>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mul_i32 s6, s15, 3                                       // 000000001608: 9606830F
	s_mov_b32 s4, s15                                          // 00000000160C: BE84000F
	s_ashr_i32 s7, s6, 31                                      // 000000001610: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001614: BF870009
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001618: 84868206
	s_waitcnt lgkmcnt(0)                                       // 00000000161C: BF89FC07
	s_add_u32 s2, s2, s6                                       // 000000001620: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001624: 82030703
	s_ashr_i32 s5, s15, 31                                     // 000000001628: 86059F0F
	s_load_b32 s6, s[2:3], null                                // 00000000162C: F4000181 F8000000
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001634: 84828204
	v_mov_b32_e32 v0, 0                                        // 000000001638: 7E000280
	s_add_u32 s0, s0, s2                                       // 00000000163C: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001640: 82010301
	s_waitcnt lgkmcnt(0)                                       // 000000001644: BF89FC07
	v_mov_b32_e32 v1, s6                                       // 000000001648: 7E020206
	global_store_b32 v0, v1, s[0:1]                            // 00000000164C: DC6A0000 00000100
	s_nop 0                                                    // 000000001654: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001658: BFB60003
	s_endpgm                                                   // 00000000165C: BFB00000
