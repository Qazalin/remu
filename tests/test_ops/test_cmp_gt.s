
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_60n2>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001614: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001618: 86039F0F
	v_mov_b32_e32 v0, 0                                        // 00000000161C: 7E000280
	s_lshl_b64 s[8:9], s[2:3], 2                               // 000000001620: 84888202
	s_waitcnt lgkmcnt(0)                                       // 000000001624: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001628: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000162C: 82070907
	s_add_u32 s0, s0, s8                                       // 000000001630: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001634: 82010901
	s_load_b32 s6, s[6:7], null                                // 000000001638: F4000183 F8000000
	s_load_b32 s0, s[0:1], null                                // 000000001640: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001648: BF89FC07
	v_cmp_lt_f32_e64 s0, s6, s0                                // 00000000164C: D4110000 00000006
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001654: BF870001
	v_cndmask_b32_e64 v1, 0, 1, s0                             // 000000001658: D5010001 00010280
	s_add_u32 s0, s4, s15                                      // 000000001660: 80000F04
	s_addc_u32 s1, s5, s3                                      // 000000001664: 82010305
	global_store_b8 v0, v1, s[0:1]                             // 000000001668: DC620000 00000100
	s_nop 0                                                    // 000000001670: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001674: BFB60003
	s_endpgm                                                   // 000000001678: BFB00000
