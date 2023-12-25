
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4n85>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	v_mov_b32_e32 v0, 0                                        // 000000001614: 7E000280
	s_ashr_i32 s3, s15, 31                                     // 000000001618: 86039F0F
	s_mov_b32 s2, s15                                          // 00000000161C: BE82000F
	s_waitcnt lgkmcnt(0)                                       // 000000001620: BF89FC07
	s_add_u32 s6, s6, s15                                      // 000000001624: 80060F06
	s_addc_u32 s7, s7, s3                                      // 000000001628: 82070307
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000162C: 84828202
	global_load_u8 v1, v0, s[6:7]                              // 000000001630: DC420000 01060000
	s_add_u32 s0, s0, s2                                       // 000000001638: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000163C: 82010301
	s_load_b32 s0, s[0:1], null                                // 000000001640: F4000000 F8000000
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001648: BF890007
	v_mul_lo_u32 v1, s0, v1                                    // 00000000164C: D72C0001 00020200
	s_add_u32 s0, s4, s2                                       // 000000001654: 80000204
	s_addc_u32 s1, s5, s3                                      // 000000001658: 82010305
	global_store_b32 v0, v1, s[0:1]                            // 00000000165C: DC6A0000 00000100
	s_nop 0                                                    // 000000001664: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001668: BFB60003
	s_endpgm                                                   // 00000000166C: BFB00000
