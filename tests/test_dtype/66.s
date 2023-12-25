
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_10n24>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_lshl_b64 s[6:7], s[4:5], 2                               // 000000001614: 84868204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_add_u32 s0, s0, s15                                      // 000000001624: 80000F00
	s_load_b32 s2, s[2:3], null                                // 000000001628: F4000081 F8000000
	s_addc_u32 s1, s1, s5                                      // 000000001630: 82010501
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	v_cmp_neq_f32_e64 s2, s2, 0                                // 000000001638: D41D0002 00010002
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001640: BF870001
	v_cndmask_b32_e64 v1, 0, 1, s2                             // 000000001644: D5010001 00090280
	global_store_b8 v0, v1, s[0:1]                             // 00000000164C: DC620000 00000100
	s_nop 0                                                    // 000000001654: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001658: BFB60003
	s_endpgm                                                   // 00000000165C: BFB00000
